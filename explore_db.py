import libsbml
import os

def leggi_tutti_sbml_in_cartella(cartella_path):
    """
    Legge tutti i file .sbml in una cartella e stampa:
    - Nome e ID del modello
    - Compartimenti
    - Specie
    - Reazioni (nome, ID, compartimento, reagenti, prodotti, reversibilitÃ)
    """
    # Scorri tutti i file nella cartella
    for filename in os.listdir(cartella_path):
        if filename.lower().endswith('.sbml'):
            file_path = os.path.join(cartella_path, filename)
            print(f"\n=== Caricando file SBML: {filename} ===")
            
            # Carica il modello
            doc = libsbml.readSBML(file_path)
            if doc.getNumErrors() > 0:
                print(f"Attenzione: ci sono {doc.getNumErrors()} errori nel file SBML")
            
            model = doc.getModel()
            if model is None:
                print("Errore: modello SBML non trovato.")
                continue
            
            print(f"\nModello: {model.getName()} (ID: {model.getId()})\n")
            
            # Compartimenti
            print("Compartimenti:")
            for comp in model.getListOfCompartments():
                print(f"  ID: {comp.getId()}, Nome: {comp.getName()}")
            print("\n")
            
            # Specie
            print("Specie:")
            for s in model.getListOfSpecies():
                print(f"  ID: {s.getId()}, Nome: {s.getName()}, Compartimento: {s.getCompartment()}")
            print("\n")
            
            # Reazioni
            print("Reazioni:")
            for reaction in model.getListOfReactions():
                print(f"Reazione: {reaction.getName()} (ID: {reaction.getId()})")
                print(f"  Compartimento: {reaction.getCompartment()}")
                
                reactants = [r.getSpecies() for r in reaction.getListOfReactants()]
                products = [p.getSpecies() for p in reaction.getListOfProducts()]
                
                print(f"  Reagenti: {reactants}")
                print(f"  Prodotti: {products}")
                print(f"  Reversibile: {reaction.getReversible()}")
                print("-----")
        return

# Esempio di utilizzo:
leggi_tutti_sbml_in_cartella("/home/pikarz/Downloads/toni/homo_sapiens.3.1.sbml/")