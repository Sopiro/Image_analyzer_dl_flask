# Machine Learning based Image Analyzing Server

Image analyzing flask web server.  
Analyze the image from three aspects. (Captioning, Text recognition, Color analyzing)  

Demo Video: https://youtu.be/bLvkysDbCZs

## Implementation

- #### Image captioning
  Faster R-CNN or Efficientnet-B4 feature extraction → Transformer network (Efficientnet-B4 extraction is used in this video demonstration)


- #### Text recognition
  Kakao OCR api (https://vision-api.kakao.com/#ocr)

- #### Color analysis
  Execute k-Means clustering on image pixels → Table matching (Color table comes from https://chir.ag/projects/name-that-color)


## Sources
Image captioning source: https://github.com/Sopiro/learning-tf2/tree/master/src/ch30_transformer_captioning

Color classification baseline: https://github.com/Sopiro/learning-tf2/tree/master/src/ch31_color_classification


## Architecture
<img src="https://raw.githubusercontent.com/Sopiro/Learning-tf2/master/src/ch30_transformer_captioning/architecture.jpg">

