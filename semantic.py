# importing spacy
import spacy

# specifying the model we want to use.
# Remember to install this model by typing
nlp = spacy.load('en_core_web_md')
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print(f"{word1}, {word2} similarity : {word1.similarity(word2)}")
print(f"{word3}, {word2} similarity : {word3.similarity(word2)}")
print(f"{word3}, {word1} similarity : {word3.similarity(word1)}")
print("-" * 50)
"""
outcome from the above similarity between cat monkey and banana 
cat, monkey similarity : 0.39452385797528866
banana, monkey similarity : 0.3741353669139763
banana, cat similarity : 0.23343780585505797

-> "Cat and monkey" have similarity as they both are  animals and this similarity is highest amongst comparision between 
"banana & monkey "  and " cat & banana".  
-> "monkey and banana" has higher similarity to "cat and banana"  as model puts "monkey and banana" together as 
monkeys eat bananas and that is why there is a significant similarity. While the model does not recognise much relationship 
between " cat and banana" 

"""
# An example of my own
word4 = nlp("Mountain")
word5 = nlp("dog")

print(f"{word1}, {word4} similarity : {word1.similarity(word4)}")      # "cat" similar to "Mountain"
print(f"{word1}, {word5} similarity : {word1.similarity(word5)}")      # "cat" similar to "dog"
print(f"{word4}, {word5} similarity : {word4.similarity(word5)}")      # "Mountain" similar to "dog"
print("-" * 50)
"""
outcome from the above similarity between cat mountain and dog 
cat, Mountain similarity : 0.4511528083668371
cat, dog similarity : 1.0000000568192473
Mountain, dog similarity : 0.4511528083668371

The highest similarity is shown between cat and dog as they both are similar kind of aminal classified under the spacy 
while the similarity of "cat & mountain" to that of "dog & mountain" is same. 
""" 

tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my car in my car",
             "I\'d like my boat back",
             "I will name my dog Diana"]
model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

print("-" * 50)