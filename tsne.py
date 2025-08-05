from models.BERT_model import BERT_model

def main():
    tsne_model = BERT_model("bert-base-uncased")
    sentences = ["Bass player likes to play bass guitar, but I prefer to fish bass in a bass pond",
                "The bass guitar is a stringed instrument that is played with the fingers or a pick",
                "Bass fishing is a popular sport in many parts of the world, especially in North America",
                "The bass is a type of fish that is found in freshwater and saltwater environments",
                "Bass players often use a variety of techniques to create different sounds and styles of music"]
    tsne_model.tSNE_plot(sentences, save_path="/home/ryba/Documents/Latex/SEM-DYP-RybinskiBartosz/img/tsne_plot.svg", show=True)
    
    
if __name__ == "__main__":
    main()