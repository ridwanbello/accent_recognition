# Data Download setup for Accent classification 

For the Afrispeech dataset on HuggingFace, `datasets==2.12.0` and `fsspec==2023.9.2` versions are recommended

## To create a dataset for classifying three region -  West Africa, East Africa, and South Africa, a sample code is shown below

``` python 

process_afrispeech_batch(accent_list = ["yoruba"], batch_name="west_africa", split="train", batch_size=1500)
process_afrispeech_batch(accent_list = ["swahili"], batch_name="east_africa", split="train", batch_size=1500)
process_afrispeech_batch(accent_list = ["afrikaans"], batch_name="south_africa", split="train", batch_size=1500)

```