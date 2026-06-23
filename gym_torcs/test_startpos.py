import xml.etree.ElementTree as ET
import time
import subprocess
import os
import snakeoil3_gym as snakeoil3

TORCS_XML = r"C:\torcs\config\raceman\practice.xml"
TORCS_DIR = r"C:\torcs"
TORCS_EXE = "wtorcs.exe"

def edit_torcs_xml(start_dist):
    tree = ET.parse(TORCS_XML)
    root = tree.getroot()
    for section in root.iter("section"):
        if section.get("name") == "Starting Grid":
            for attnum in section.iter("attnum"):
                if attnum.get("name") == "distance to start":
                    attnum.set("val", str(float(start_dist)))
                    tree.write(TORCS_XML, encoding="UTF-8", xml_declaration=True)
                    return
    raise ValueError("Nie znaleziono pola")

def restart_torcs():
    subprocess.call(["taskkill", "/f", "/im", TORCS_EXE],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    subprocess.Popen(
        [os.path.join(TORCS_DIR, TORCS_EXE), "-nofuel", "-nodamage", "-nolaptime", "-port", "3001"],
        cwd=TORCS_DIR
    )
    time.sleep(3.0)

for test_val in [0, 100, 500, 1000]:
    edit_torcs_xml(test_val)
    restart_torcs()
    client = snakeoil3.Client(p=3001)
    client.get_servers_input()
    print(f"val={test_val} → distFromStart={client.S.d['distFromStart']:.1f}")
    client.shutdown()