import spacy

# load the model
nlp = spacy.load('en_core_web_md')


# Returns the name of a movie from 'movies.txt' where description is most similar to given description to be compared

def watch_next(desc):
    # read the movies.txt file with description
    with open("movies.txt", "r", encoding="utf-8") as f:
        movie_data = f.read().split("\n")

    # Dict to hold movie name  and similarity scores
    movie_name_sc = {}

    for index, movie in enumerate(movie_data):

        # ignoring any blank line in the movie data
        if movie.strip() == "":
            movie_data.pop(index)

        # Perform similarity check
        else:
            name = movie.split(" :")[0]
            movie_desc = movie.split(" :")[1]
            movie_name_sc[name] = nlp(movie_desc).similarity(nlp(desc))

    for movie in movie_name_sc:
        if movie_name_sc[movie] == max(movie_name_sc.values()):
            return movie


desc_to_comp = "Will he save their world or destroy it? When the Hulk becomes too dangerous for the Earth, the Illuminati trick Hulk into a shuttle and launch him into space to a planet where the Hulk can live in peace. Unfortunately, Hulk land on the planet Sakaar where he is sold into slavery and trained as a gladiator."

next_movie = watch_next(desc_to_comp)

print(f"System recommended next movie to watch  '{next_movie}'")
