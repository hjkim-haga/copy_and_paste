import sys
import os
import xml.etree.ElementTree as ET

working_dir = '/home/ubuntu/haga-dataset/electronics/valid_batch_from_center/xmls/'

for xml_path in [os.path.join(working_dir, xmlfile) for xmlfile
                 in os.listdir(working_dir)]:
    filename =os.path.basename(xml_path)
    print(filename)


    tree = ET.parse(os.path.abspath(xml_path))
    root = tree.getroot()
    filename_tag = root.find('filename')
    filename_tag.text = os.path.splitext(filename)[0] + '.jpg'
    
    tree.write(xml_path)


    