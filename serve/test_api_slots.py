"""Request ownership and raw tokenizer contracts; no GPU or model process."""
import json,threading,unittest,urllib.request,urllib.error
from pathlib import Path
from types import SimpleNamespace
from serve.server import (SlotLogprobs,request_lp_lines,LogprobEntries,CompletionLogprobs,
                         Service,MockEngine,ByteTokenizer,serve,check_logprobs)
from serve.frontend import ChatTemplate
from serve.test_logprobs import Tokenizer

class Routing(unittest.TestCase):
    def test_interleaved_slots_deliver_probabilities_before_their_token(self):
        a,b=threading.Event(),threading.Event();qa,qb=request_lp_lines(a),request_lp_lines(b)
        r=SlotLogprobs();r.slots={0:qa,1:qb}
        source=['BT 0 65\n','BT 1 66\n','BLP 1 -0.2 65:-2\n','BLP 0 -0.1 66:-3\n','BDONE 0 1 stop 1\n']
        it=r.feed(source);self.assertEqual(next(it),'BT 1 66\n');self.assertEqual(qb[0],(66,'LP -0.2 65:-2'))
        self.assertEqual(next(it),'BT 0 65\n');self.assertEqual(qa[0],(65,'LP -0.1 66:-3'))
        self.assertEqual(list(it),['BDONE 0 1 stop 1\n']);self.assertNotIn(0,r.slots)
    def test_solo_to_slot_metadata_stays_on_one_request(self):
        e=threading.Event();q=request_lp_lines(e);r=SlotLogprobs();r.control=q
        self.assertEqual(list(r.feed(['T 65\n','LP -0.1 66:-2\n','DONE 1\n'])),['T 65\n','DONE 1\n'])
        r.slots[1]=q;list(r.feed(['BT 1 66\n','BLP 1 -0.2 65:-3\n','BDONE 1 1 stop 1\n']))
        self.assertEqual([t for t,_ in q],[65,66])
    def test_parser_tail_survives_other_request_replacing_global_lines(self):
        tok=Tokenizer();a,b=threading.Event(),threading.Event();svc=SimpleNamespace(tok=tok,stop_ids=[])
        parser=LogprobEntries(svc,owner=a);qa=request_lp_lines(a);qa.append((65,'LP -0.1 66:-2'))
        eng=SimpleNamespace(lp_lines=request_lp_lines(b));eng.lp_lines.append((66,'LP -0.8 65:-1'))
        self.assertEqual(parser.spell(eng,'A')[0]['token'],'A');self.assertEqual(len(eng.lp_lines),1)
    def test_completion_reader_keeps_original_sink_after_slot_reuse(self):
        tok=Tokenizer();a,b=threading.Event(),threading.Event();svc=SimpleNamespace(tok=tok,stop_ids=[])
        p=CompletionLogprobs(svc,{},a);qa=request_lp_lines(a);qa.append((65,'LP -0.1 66:-2'))
        p.watch(SimpleNamespace(lp_lines=request_lp_lines(b)))
        self.assertEqual(p.build(0)['tokens'],['A'])
    def test_plain_and_unsupported_serial_protocol_pass_through(self):
        lines=['T 65\n','LP -0.1\n','BT 0 66\n','BDONE 0 1 stop 0\n','DONE 1\n'];self.assertEqual(list(SlotLogprobs().feed(lines)),lines)
    def test_missing_metadata_is_not_fabricated(self):
        q=request_lp_lines(threading.Event());r=SlotLogprobs();r.slots[0]=q
        lines=['BT 0 65\n','BDONE 0 1 stop 0\n'];self.assertEqual(list(r.feed(lines)),lines);self.assertEqual(list(q),[])
    def test_slot_capability_gate(self):
        svc=SimpleNamespace(engine=SimpleNamespace(batch=2,info={'logprobs':1,'slot_logprobs':1}),tok=SimpleNamespace(tokens=list(range(100))))
        check_logprobs({'logprobs':True},svc,16)
        svc.engine.info={}
        with self.assertRaises(ValueError):check_logprobs({'logprobs':True},svc,16)

class Tokenize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok=ByteTokenizer();cls.eng=MockEngine(cls.tok,'A',max_context=4096)
        cls.svc=Service(cls.eng,cls.tok,ChatTemplate(Path(__file__).parent/'chat_template.jinja'));cls.svc.api_key='test-only'
        cls.http=serve(cls.svc,port=0);cls.base='http://127.0.0.1:'+str(cls.http.server_address[1])
    @classmethod
    def tearDownClass(cls):cls.http.shutdown();cls.http.server_close()
    def post(self,body,key='test-only'):
        req=urllib.request.Request(self.base+'/tokenize',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
        try:
            with urllib.request.urlopen(req) as r:return r.status,json.load(r)
        except urllib.error.HTTPError as e:return e.code,json.load(e)
    def test_cyrillic_ids_match_the_server_tokenizer_without_chat_template(self):
        text='Привет, мир!';code,r=self.post({'content':text,'add_special':False});self.assertEqual(code,200);self.assertEqual(r['tokens'],self.tok.encode(text,parse_special=True));self.assertEqual(r['count'],len(r['tokens']))
    def test_empty_input(self):self.assertEqual(self.post({'content':''}), (200,{'tokens':[],'count':0}))
    def test_with_pieces(self):
        code,r=self.post({'content':'AB','with_pieces':True});self.assertEqual(code,200);self.assertEqual(r['tokens'],[{'id':65,'piece':'A'},{'id':66,'piece':'B'}])
    def test_authentication_required(self):self.assertEqual(self.post({'content':'secret'},'wrong')[0],401)
    def test_invalid_payload(self):
        for b in [{'content':42},{'content':'a','prompt':'b'},{'content':'a','with_pieces':1}]:
            with self.subTest(b=b):self.assertEqual(self.post(b)[0],400)

class ProbabilityLimits(unittest.TestCase):
    def setUp(self):
        self.svc=SimpleNamespace(engine=SimpleNamespace(batch=0,info={'logprobs':1}),tok=SimpleNamespace(tokens=list(range(1000))))
    def test_older_serial_engine_is_rejected(self):
        self.svc.engine.info={}
        with self.assertRaisesRegex(ValueError,'INFO logprobs'):check_logprobs({'logprobs':True},self.svc,1)
    def test_allowed_ids_are_unique_and_bounded(self):
        from serve.server import allowed_ids
        self.assertEqual(allowed_ids({'allowed_token_ids':[65,66,66]}),[65,66])
        for ids in ([65]*257,[-1],[True],'65', [1.5]):
            with self.subTest(ids=str(ids)[:32]),self.assertRaises(ValueError):check_logprobs({'logprobs':True,'allowed_token_ids':ids},self.svc,1)
    def test_total_metadata_is_bounded(self):
        req={'logprobs':True,'allowed_token_ids':list(range(256))}
        check_logprobs(req,self.svc,512)
        with self.assertRaisesRegex(ValueError,'metadata'):check_logprobs(req,self.svc,513)
    def test_refused_admission_cannot_reassign_previous_slot_owner(self):
        a,b=threading.Event(),threading.Event();qa,qb=request_lp_lines(a),request_lp_lines(b)
        r=SlotLogprobs();r.slots[0]=qa;r.admission=(0,qb)
        list(r.feed(['BT 0 65\n','BLP 0 -0.1\n','ERR BGEN: no such free slot\n']))
        self.assertEqual(list(qa),[(65,'LP -0.1')]);self.assertEqual(list(qb),[])
        self.assertIs(r.slots[0],qa);self.assertIsNone(r.admission)
    def test_successful_admission_changes_owner_at_native_acknowledgement(self):
        a,b=threading.Event(),threading.Event();qa,qb=request_lp_lines(a),request_lp_lines(b)
        r=SlotLogprobs();r.slots[0]=qa;r.control=qb;r.admission=(0,qb)
        list(r.feed(['T 66\n','LP -0.2\n','DONE 1\n','BADM 0 1\n','BT 0 67\n','BLP 0 -0.3\n']))
        self.assertEqual([t for t,_ in qb],[66,67]);self.assertEqual(list(qa),[])

class RawCompletionContract(unittest.TestCase):
    def setUp(self):
        from serve.test_logprobs import ThinkTokenizer,LpEngine,TEMPLATE
        self.text='<think>проверка</think><tool_call>literal</tool_call>ответ'
        self.tok=ThinkTokenizer();self.eng=LpEngine(self.tok,self.text)
        self.svc=Service(self.eng,self.tok,TEMPLATE)
        # These chat conveniences must never rewrite a raw completion.
        self.svc.answer_room=True;self.svc.reasoning_close_retry=True
        self.http=serve(self.svc,port=0);self.base='http://127.0.0.1:'+str(self.http.server_address[1])
    def tearDown(self):self.http.shutdown();self.http.server_close()
    def post(self,body):
        req=urllib.request.Request(self.base+'/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req) as r:return r.status,json.load(r)
        except urllib.error.HTTPError as e:return e.code,json.load(e)
    def test_raw_markers_and_token_ids_are_preserved(self):
        ids=self.tok.encode('Прямой промпт',parse_special=True)
        code,r=self.post({'prompt':ids,'enable_thinking':False,'max_tokens':256,'logprobs':3})
        self.assertEqual(code,200);self.assertEqual(self.eng.last_prompt,ids)
        self.assertEqual(r['choices'][0]['text'],self.text)
        # Byte-split Cyrillic uses offsets of the text actually decoded, not U+FFFD per byte.
        lp=r['choices'][0]['logprobs'];offset=0;decoder=__import__('codecs').getincrementaldecoder('utf-8')('replace')
        for t,pos in zip(self.eng.script,lp['text_offset']):
            self.assertEqual(pos,offset)
            if t not in self.svc.stop_ids:offset+=len(decoder.decode(self.tok.token_bytes(t)))
    def test_draining_and_queue_limit_apply_to_raw_completions(self):
        self.svc.draining=True
        code,r=self.post({'prompt':'x','max_tokens':1});self.assertEqual(code,503);self.assertEqual(r['error']['code'],'server_draining')
        self.svc.draining=False;self.svc.max_queue=0;self.svc.admission_reserved=self.svc.admission_limit()
        code,r=self.post({'prompt':'x','max_tokens':1});self.assertEqual(code,429);self.assertEqual(r['error']['code'],'admission_full')
    def test_stop_string_cuts_raw_output_without_chat_parsing(self):
        code,r=self.post({'prompt':'x','max_tokens':256,'stop':'</think>'})
        self.assertEqual(code,200);self.assertEqual(r['choices'][0]['text'],'<think>проверка')
    def test_unsupported_serial_native_capability_is_a_400(self):
        self.eng.info={}
        code,r=self.post({'prompt':'x','max_tokens':1,'logprobs':1})
        self.assertEqual(code,400);self.assertIn('INFO logprobs',r['error']['message'])
