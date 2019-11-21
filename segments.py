vowels = set(('æ', 'ɔ', 'ɛ', 'e', 'I'))
consonants = set(('k', 'g', 'd', 'h', 't', 'ʔ', 'w',
                  's', 'z', 'n', 'l', 'v', 'f', 'θ'))
phones = vowels.union(consonants)
voiced_consonants = set(('d', 'w', 'g', 'z', 'v', 'n'))
voiced = vowels.union(voiced_consonants)
voiceless_consonants = consonants - voiced_consonants
sonorants = vowels.union(set(('l', 'n', 'w')))
obstruents = phones - sonorants
voice_obstruents = obstruents - voiceless_consonants
voiceless_obstruents = obstruents - voice_obstruents
