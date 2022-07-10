from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import os

ROOT_FOLDER = os.path.dirname(__file__)

for i in [
    '\u6b69', '\u9999', '\u6842', '\u9280', '\u91d1', '\u89d2', '\u98db',
    '\u7389', '\u3068', '\u674f', '\u572d', '\u5168', '\u99ac', '\u9f8d'
]:
    img = Image.new('RGB', (90,90), color=(255, 234, 176,0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(ROOT_FOLDER+"/MaShanZheng-Regular.ttf", 70)
    #font = ImageFont.truetype(ROOT_FOLDER+"/CODE2000.ttf", 70)
    draw.text((10, 10),i,(0,0,0),font=font)
    img.save('sample-out'+i+'.jpg')
            
for i in [
    '\u6b69', '\u9999', '\u6842', '\u9280', '\u91d1', '\u89d2', '\u98db',
    '\u7389', '\u3068', '\u674f', '\u572d', '\u5168', '\u99ac', '\u9f8d'
]:
    img = Image.new('RGB', (90,90), color=(255, 234, 176,0))
    draw = ImageDraw.Draw(img)
    #font = ImageFont.truetype(ROOT_FOLDER+"/MaShanZheng-Regular.ttf", 70)
    font = ImageFont.truetype(ROOT_FOLDER+"/CODE2000.ttf", 70)
    draw.text((10, 10),i,(0,0,0),font=font)
    img.save('sample-out'+i[]+'.jpg')