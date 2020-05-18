import subprocess
import os
import pdb


class Abbr_resolver():

    def __init__(self, ab3p_path):
        self.ab3p_path = ab3p_path
        
    def resolve(self, corpus_path):
        result = subprocess.run([self.ab3p_path, corpus_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        line = result.stdout.decode('utf-8')
        if "Path file for type cshset does not exist!" in line:
            raise Exception(line)
        elif "Cannot open" in line:
            raise Exception(line)
        elif "failed to open" in line:
            raise Exception(line)
        lines = line.split("\n")
        result = {}
        for line in lines:
            if len(line.split("|"))==3:
                sf, lf, _ = line.split("|")
                sf = sf.strip()
                lf = lf.strip()
                result[sf] = lf
        
        return result