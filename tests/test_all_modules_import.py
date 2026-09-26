import importlib, pkgutil
import kfc_procedure

def test_import_every_module():
    failures=[]
    for m in pkgutil.walk_packages(kfc_procedure.__path__, kfc_procedure.__name__+'.'):
        try: importlib.import_module(m.name)
        except Exception as e: failures.append((m.name,repr(e)))
    assert not failures, failures
