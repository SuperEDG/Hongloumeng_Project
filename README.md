# Dream of the Red Chamber Chatbot Project

![LCTs](https://github.com/SuperEDG/Hongloumeng_Project/assets/14871187/eaca7916-b1a3-4d0a-8653-6e7b5e28eedf)
> Example of the character Lin Daiyu's personality traits from《红楼梦》(Dream of the Red Chamber) and associated dialogues. On the left, Lin Daiyu's personality traits are displayed. On the right, representative conversations that reflect these personality traits are presented.

This repository provides a source code for paper [Dialogue Agents with Literary Character Personality Traits](https://eprints.ncl.ac.uk/293995), including datasets and models utilized in studies.
This project aims to enhance chatbot engagement by leveraging unique personalities and speech patterns extracted from character dialogues in literary works. 

We utilized 《红楼梦》(Dream of the Red Chamber) as a data source, developed an automated system for efficient extraction and mapping of character dialogues and personality traits. 
This approach results in chatbots more resembling human conversation, proven by evaluations. 

## Repository Structure

The repository consists of three main directories: `data`, `model`, and `process`.

### Data

- `resources`: Contains the original data.
- `processed`: Contains the processed data.

### Process

- `chinese_traits_extractor.py`: Builds the Chinese traits word library.
- `clean_and_store_reviews.py`: Cleans the book's reviews.
- `extractDialogues.py`: Extracts dialogues from the book.

### Model

- `2-NER`: Contains the Named Entity Recognition model for extracting character names.
- `3-LCPM`: Constructs the Literary Character Personality Map (LCPM) and selects suitable characters.
- `4-DialogueAgent`: Builds the final chatbot model using the personality maps and dialogues.

## Replicating the Project

To use this system with a new book:

1. **Book Addition**: Add the new book to `data/resources/books`.

2. **Dialogue Extraction**: Run `extractDialogues.py` in the `process` directory. Extracted dialogues will be stored in `data/processed/books`.

3. **Dialogue and Character Extraction**: Run `extract_dialogues.py` in `model/2-NER`. This will extract dialogues and character names.

4. **Character Selection**: 
   - First, search for corresponding character reviews and store them in `data/resources/Reviews`.
   - Then run `clean_and_store_reviews.py` in the `process` directory. The cleaned reviews will be stored in `data/processed/LCPM`.
   - Run `generateLCPM.py` in `model/3-LCPM` to create the character personality map.
   - Run `tsne_characters.py` for visualizing character personalities, and `character_selection.py` to select suitable characters.

5. **Chatbot Model Construction**: Use the constructed dataset to build a generative model in `model/4-DialogueAgent`.

## References

```bibtex
@inproceedings{zhao2023dialogue,
  title={Dialogue Agents with Literary Character Personality Traits},
  author={Zhao, Junzhe and Liang, Huizhi and Rusnachenko, Nicolay},
  booktitle={The 22nd IEEE/WIC International Conference on Web Intelligence and Intelligent Agent Technology},
  year={2023},
  organization={Newcastle University}
}
```
