#************* HELP *****************
#REMEMBER THAT IF YOU NEED SUPPORT ON ANY ASPECT OF YOUR COURSE SIMPLY LOG IN TO www.hyperiondev.com/support TO:
#START A CHAT WITH YOUR MENTOR, SCHEDULE A CALL OR GET SUPPORT OVER EMAIL.
#*************************************

# *** NOTE ON COMMENTS ***
# This is a comment in Python.
# Comments can be placed anywhere in Python code and the computer ignores them - they are intended to be read by humans.
# Any line with a # in front of it is a comment and any with  ''' is also a docstring.
# Please read all the comments in this example file and all others.

import spacy  # importing spacy
nlp = spacy.load('en_core_web_sm') # specifying the model we want to use. Remember to install this model by typing python -m spacy download en_core_web_md into your command line

# Now we are going to look into longer texts and compare them.
# Below we  have two lists: one containing complaints submitted to a company, and another of recipes found online.
# We want to establish how spaCy's model can identify similarities or dissimilarities between complaint and recipes.

# Make sure to run this example file and read through the explanations.

# Below is a list of six complaints.
complaints = [ 'We bought a house in  CA. Our mortgage was handled by a company called ki. Soon after the mortgage was sold to ABC. Shortly after that XYZ took over the mortgage. The other day we got a notice not to send our payment to them but to loi instead. This is all so frustrating and wreaks of the  mortgage nightmare.',
'I got approved for a loan to buy a house I have submitted everything I need to for them I paid for the inspection and paid good faith check after all of that they said I did not get approved for the loan to cancel my contract because they do not want to wait for the down payments assistant said that the Sellers do not want to wait that long I feel like they are getting over on me I feel that they should have told me that I did not get approved before I spent my money and picked out a house Carrington mortgage in Ohio ',
'As per the correspondence, I received from : The University  This is to inform you that I have recently pulled my credit report and noticed that there is a collection listing from The University  on my credit report. I WAS never notified of this collection action or that I owed the debt. This letter is to inform you that I would like a verification of the debt and juilo ability to collect this money from me.',
'I am writing to dispute the follow information in my file.ON BOTH TransUnion & . for {$15000.00}. I have contacted this agency to advise to STOP CALLING ME this case was dismissed in court  2014. Please see the attached document from  County State Court. Thanking you in advanced regarding this matter.',
'I have not had a XXXX phone since early 2007. I have tried to resolve my bill in the past but it keeps reposting an old bill. I have no way to provide financial info from 8 years ago and they know that so they want me to prove it to them but I have no way to do that. Is there anyway to get  to find out how old it is.',
'I posted dated a check and mailed it for 2015 for my mortgage payment as my mortgage company will only take online payments if all the late charges are paid at once ( also illegal ), and the check was cashed on 2015 which cost me over {$70.00} in over draft fees with my bank.'
]

# We will now compare the similarity of the complaints to ascertain if spaCy's similarity
# model is able to distinguish between these long pieces of text.

print("-------------Complaints similarity---------------")
complaint_similarity = []
for token in complaints:
    token = nlp(token)
    for token_ in complaints:
        token_ = nlp(token_)
        print(token.similarity(token_))
        complaint_similarity.append(token.similarity(token_))

print(f"\nAverage value = {sum(complaint_similarity) / len(complaint_similarity)}\n")
# Below is a list of six recipe instructions.

recipes= [ 'Bake in the preheated oven, stirring every 20 minutes, until sugar mixture has baked and caramelized onto popcorn and cashews, about 1 hour. Spread cashew caramel corn onto a parchment paper-lined baking sheet to cool. If desired, form into balls while still warm.',
'Combine brown sugar, corn syrup, butter, salt, and cream of tartar in a large saucepan. Bring to a boil, stirring constantly, until a candy thermometer inserted into the middle of the syrup, not touching the bottom, reads 260 degrees F (127 degrees C), 6 to 8 minutes.',
'Lift marshmallow fudge out of the pan by the edges of the foil and place on a large cutting board. Dip a large knife in the remaining confectioners\' sugar and slice fudge into 1 1/2-inch squares, continually dipping knife in the sugar after each slice.',
'Melt butter in a medium saucepan over medium heat; stir in condensed milk. Pour in chocolate chips; cook and stir until melted, 5 to 10 minutes.',
'Lightly grease a cookie sheet. Deflate the dough and turn it out onto a lightly floured surface. Roll the marzipan into a rope and place it in the center of the dough. Fold the dough over to cover it; pinch the seams together to seal. Place the loaf, seam side down, on the prepared baking sheet. Cover with a damp cloth and let rise until doubled in volume, about 40 minutes. Meanwhile, preheat oven to 350 degrees F (175 degrees C)',
'In a large bowl, cream together the butter, brown sugar, and white sugar. Beat in the instant pudding mix until blended. Stir in the eggs and vanilla. Blend in the flour mixture. Finally, stir in the chocolate chips and nuts. Drop cookies by rounded spoonfuls onto ungreased cookie sheets.'
]

# We will now compare the similarity of the recipes. to ascertain how well spaCy's similarity
# model is able to distinguish between them.

print("-------------Recipes similarity---------------")
recipe_similarity = []
for token in recipes:
    token = nlp(token)
    for token_ in recipes:
        token_ = nlp(token_)
        print(token.similarity(token_))
        recipe_similarity.append(token.similarity(token_))

print(f"\nAverage value = {sum(recipe_similarity) / len(recipe_similarity)}\n")

# Now we want to obtain the extent of similarity between the complaints and the recipes.
# we will loop through every recipe instruction and compare it with a complaint.

print("-------------Recipes & Complaints similarity---------------")
recipe_and_complaints =[]
for token in recipes:
    token = nlp(token)
    for token_ in complaints:
        token_ = nlp(token_)
        print(token.similarity(token_))
        recipe_and_complaints.append(token.similarity(token_))

print(f"\nAverage value = {sum(recipe_and_complaints) / len(recipe_and_complaints)}")
# What do you observe? Note that the similarity index has reduced from what we observed in the short-text example discussed in the content PDF.


# There are several ways to make your model more accurate with the similarity
# or even prediction such as feeding it with some training data. This could include
# more vocabulary about food and recipes if you are building a models concerning food.
# You can also head over to spaCy documentation here: https://spacy.io/usage/vectors-similarity
# and check out other cool stuff!

'''
executed example.py using the 'en_core_web_sm'  model  instead of 'en_core_web_md' , below results were found out 
using en_core_web_md                                using en_core_web_sm
-------------Complaints similarity---------------	-------------Complaints similarity---------------
1  	                                                1
0.957191988	                                        0.502017375
0.974148122	                                        0.730700227
0.959226772	                                        0.766111205
0.947110577	                                        0.636312218
0.936448924	                                        0.697317804
0.957191988	                                        0.502017375
1	                                                1
0.974561331	                                        0.734349273
0.947545299	                                        0.522660475
0.975974574	                                        0.788966097
0.921967088	                                        0.539953883
0.974148122	                                        0.730700227
0.974561331	                                        0.734349273
1	                                                1
0.971613744	                                        0.722292888
0.961588174	                                        0.682499839
0.949103268	                                        0.654786785
0.959226772	                                        0.766111205
0.947545299	                                        0.522660475
0.971613744	                                        0.722292888
1	                                                1
0.939283079	                                        0.635773385
0.938824066	                                        0.568389936
0.947110577	                                        0.636312218
0.975974574	                                        0.788966097
0.961588174	                                        0.682499839
0.939283079	                                        0.635773385
1	                                                1
0.878628175	                                        0.533125324
0.936448924	                                        0.697317804
0.921967088	                                        0.539953883
0.949103268	                                        0.654786785
0.938824066	                                        0.568389936
0.878628175	                                        0.533125324
1	                                                1
-------------Recipes similarity---------------	-------------Recipes similarity---------------
1	                                                1
0.945670441	                                        0.750477924
0.919843649	                                        0.630347481
0.918922195	                                        0.818663337
0.927322428	                                        0.746251583
0.938239303	                                        0.730980594
0.945670441	                                        0.750477924
1	                                                1
0.947078235	                                        0.647898004
0.933188385	                                        0.773601535
0.960729597	                                        0.664975458
0.954794576	                                        0.685839978
0.919843649	                                        0.630347481
0.947078235	                                        0.647898004
1	                                                1
0.906433384	                                        0.610521812
0.958589682	                                        0.72743346
0.94234408	                                        0.698956674
0.918922195	                                        0.818663337
0.933188385	                                        0.773601535
0.906433384	                                        0.610521812
1	                                                1
0.875873528	                                        0.717782589
0.920736621	                                        0.751233151
0.927322428	                                        0.746251583
0.960729597	                                        0.664975458
0.958589682	                                        0.72743346
0.875873528	                                        0.717782589
1	                                                1
0.93515382	                                        0.717308137
0.938239303	                                        0.730980594
0.954794576	                                        0.685839978
0.94234408	                                        0.698956674
0.920736621	                                        0.751233151
0.93515382	                                        0.717308137
1	1
-------------Recipes similarity against complaints --------------
0.8028610287478102	                                0.652935699
0.818396148	                                        0.284429439
0.811030614	                                        0.451458839
0.78589022	                                        0.679826398
0.820493415	                                        0.411856781
0.762080515	                                        0.697261347
0.849164143	                                        0.511398613
0.831886504	                                        0.141312204
0.846553516	                                        0.390253562
0.83594031	                                        0.550159904
0.83207185	                                        0.219419867
0.818230453	                                        0.581529206
0.823434613	                                        0.588183369
0.791729708	                                        0.262907292
0.814448006	                                        0.517455263
0.798632603	                                        0.602576202
0.7648883	                                        0.329785769
0.794077472	                                        0.586716288
0.687085516	                                        0.493490218
0.672986033	                                        0.184936894
0.67635467	                                        0.351749065
0.669645114	                                        0.564966041
0.664977814	                                        0.297707869
0.666869734	                                        0.585533782
0.894571719	                                        0.742299793
0.869301281	                                        0.385087257
0.889839381	                                        0.568350593
0.880377316	                                        0.746489232
0.856566896	                                        0.567362784
0.860298875	                                        0.644071301
0.803699819	                                        0.603089987
0.782875267	                                        0.088030825
0.793703381	                                        0.4390962
0.768746081	                                        0.62641398
0.772643965	                                        0.279210732
0.744652633	                                        0.463810033


There is high difference between the similarity obtained from these 2 models.
Similarity obtained from model en_core_web_md is better than similarity obtained from model en_core_web_sm
There is also a  user warning that there are no word vectors provided with that language model.
I did try to google it but didnt find any relevant information and nothing is mentioned in the course material , thus no comment on this 
'''