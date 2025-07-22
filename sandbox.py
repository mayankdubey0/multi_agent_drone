import mujoco
from mujoco import viewer

model = mujoco.MjModel.from_xml_path(r"C:\mujoco-3.3.2-windows-x86_64\model\skydio_x2\x2.xml")

# model = mujoco.MjModel.from_xml_string(XML)
data  = mujoco.MjData(model)
viewer = viewer.launch(model, data)

while True:
    mujoco.mj_step(model, data)
    # print only when rope is taut
    
