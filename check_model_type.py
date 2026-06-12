"""Check if a .keras file is a ProgressiveDistiller wrapper or standalone model."""
import zipfile, json, sys

p = sys.argv[1]
with zipfile.ZipFile(p) as z:
    cfg = json.loads(z.read("config.json"))
    
print(f"Class: {cfg.get('class_name', '?')}")
print(f"Name: {cfg.get('config', {}).get('name', '?')}")

# Check if it contains a ProgressiveDistiller
config_str = json.dumps(cfg).lower()
if "progressivedistiller" in config_str:
    print("Type: ProgressiveDistiller wrapper")
elif "distiller" in config_str:
    print("Type: Distiller wrapper") 
else:
    print("Type: Standalone model (no distiller)")