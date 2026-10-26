"""Assemble pinned application/native sources without patch application."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,tarfile,shutil
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--work-dir',required=True,type=Path)
p.add_argument('--prepare-only',action='store_true')
a=p.parse_args();m=json.loads((ROOT/'local/components.lock.json').read_text())
subprocess.run(['python3',str(ROOT/'local/verify-source.py')],check=True)
subprocess.run(['git','-C',str(ROOT),'cat-file','-e',m['engine_commit']+'^{commit}'],check=True)
work=a.work_dir.resolve();work.mkdir(parents=True,exist_ok=False)
def archive(commit,target,paths=()):
 target.mkdir();temp=work/(target.name+'.tar')
 with temp.open('wb') as f:subprocess.run(['git','-C',str(ROOT),'archive','--format=tar',commit,*paths],stdout=f,check=True)
 with tarfile.open(temp) as f:f.extractall(target,filter='data')
 temp.unlink()
archive(m['engine_commit'],work/'engine-source')
archive('HEAD',work/'application-source',('serve','tools','data','strata_cli','strata_control','strata.py','setup.py','requirements.txt'))
for f,h in m['engine_product_files'].items():
 if hashlib.sha256((work/'engine-source'/f).read_bytes()).hexdigest()!=h:raise SystemExit('Engine source mismatch: '+f)
app=work/'application-source';(app/'VERSION').write_text(m['control_version']+'\n')
release={**m,'application_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'prepared_only':a.prepare_only}
if not a.prepare_only:
 env=dict(os.environ,BUILD_DIR=str(work/'native-build'))
 subprocess.run(['bash',str(work/'engine-source/local/build-native.sh')],env=env,check=True)
 (work/'artifacts').mkdir();shutil.copy2(work/'native-build/strata',work/'artifacts/strata')
 release['binary_sha256']=hashlib.sha256((work/'artifacts/strata').read_bytes()).hexdigest()
(work/'release.json').write_text(json.dumps(release,indent=2)+'\n')
print('PASS pinned source assembly'+(' (no native build/GPU/deployment)' if a.prepare_only else ' and linked native build'))
