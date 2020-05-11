import subprocess
import os


class Abbr_resolver():

    def __init__(self, ab3p_path):
        self.ab3p_path = ab3p_path
        
    def resolve(self, corpus_path):
        result = subprocess.run([self.ab3p_path, corpus_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        line = result.stdout.decode('utf-8')
        error = result.stderr.decode('utf-8')
        if error == "Path file for type cshset does not exist!":
            raise "Path file for type cshset does not exist!"
        elif "Cannot open" in error:
            raise "Cannot open file"
        lines = line.split("\n")
        result = {}
        for line in lines:
            if len(line.split("|"))==3:
                sf, lf, _ = line.split("|")
                sf = sf.strip()
                lf = lf.strip()
                result[sf] = lf
        
        return result