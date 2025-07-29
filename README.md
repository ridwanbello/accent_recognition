# accent_recognition
Accent Recognition

### To download any file from GitHub on Google Colab, use:

``` python

try:
    from data_download.afrispeech_data_setup import process_afrispeech_batch`
except:
   
    !git clone https://github.com/ridwanbello/accent_recognition
    !mv accent_recognition/data_download .
    !rm -rf accent_recognition
    from data_download.afrispeech_data_setup import process_afrispeech_batch

```