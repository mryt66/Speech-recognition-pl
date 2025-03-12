# Speech-recognition-for-Polish-language
Purpose of this project is to fit as small as possible model into microcomuputer to make it inference locally.

To find and fix possible issues, program uses python library called ```language_tool_python```

For now there are 2 versions of tunned models.
- whisper-tiny-pl
- whisper-base-pl

You can test models on huggingface spaces:
- https://huggingface.co/spaces/marcsixtysix/whisper-tiny-pl-tunned
- https://huggingface.co/spaces/marcsixtysix/Speech-recognition-pl-small

Metrics for specified models tested on polish language from common_voice set (over 700 validated examples):
<p align="center">
  <img src="https://github.com/user-attachments/assets/cda22cf2-1a35-431f-87f9-d73c7086f0d0" />
  <br />
</p>



Here are also tests for comparision for bigger models.
<p align="center">
  <img src="https://github.com/user-attachments/assets/d7d80dee-3f81-4c38-85b3-0f8ab0c46733" />
  <br />
</p>


