# Each accents in the Afrispeech dataset can be fetched by running:

`process_afrispeech_batch(accent_list = ["yoruba"], batch_name="west_africa", split="train", batch_size=1500)` (for Yoruba and West Africa)
`process_afrispeech_batch(accent_list = ["swahili"], batch_name="east_africa", split="train", batch_size=1500)` (for Swahili and East Africa)
`process_afrispeech_batch(accent_list = ["afrikaans"], batch_name="south_africa", split="train", batch_size=1500)` (for Afrikaans and South Africa)