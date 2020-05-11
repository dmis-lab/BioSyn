import re
from string import punctuation


class TextPreprocess():
    """
    Text Preprocess module
    Support lowercase, removing punctuation, typo correction
    """
    def __init__(self, 
            lowercase=True, 
            remove_punctuation=True,
            ignore_punctuations="",
            typo_path=None):
        """
        Parameters
        ==========
        typo_path : str
            path of known typo dictionary
        """
        self.lowercase = lowercase
        self.typo_path = typo_path
        self.rmv_puncts = remove_punctuation
        self.punctuation = punctuation
        for ig_punc in ignore_punctuations:
            self.punctuation = self.punctuation.replace(ig_punc,"")
        self.rmv_puncts_regex = re.compile(r'[\s{}]+'.format(re.escape(self.punctuation)))
        
        if typo_path:
            self.typo2correction = self.load_typo2correction(typo_path)
        else:
            self.typo2correction = {}

    def load_typo2correction(self, typo_path):
        typo2correction = {}
        with open(typo_path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                s = line.strip()
                tokens = s.split("||")
                value = "" if len(tokens) == 1 else tokens[1]
                typo2correction[tokens[0]] = value    

        return typo2correction 

    def remove_punctuation(self,phrase):
        phrase = self.rmv_puncts_regex.split(phrase)
        phrase = ' '.join(phrase).strip()

        return phrase

    def correct_spelling(self, phrase):
        phrase_tokens = phrase.split()
        phrase = ""

        for phrase_token in phrase_tokens:
            if phrase_token in self.typo2correction.keys():
                phrase_token = self.typo2correction[phrase_token]
            phrase += phrase_token + " "
       
        phrase = phrase.strip()
        return phrase

    def run(self, text):
        if self.lowercase:
            text = text.lower()

        if self.typo_path:
            text = self.correct_spelling(text)

        if self.rmv_puncts:
            text = self.remove_punctuation(text)

        text = text.strip()

        return text