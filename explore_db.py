import libsbml
import os

def leggi_sbml_file(file_path):
    """
    Legge un file .sbml e stampa:
    - Nome e ID del modello
    - Compartimenti
    - Specie
    - Reazioni (nome, ID, compartimento, reagenti, prodotti, reversibilitÃ)
    """
    filename = os.path.basename(file_path)
    if not filename.lower().endswith('.sbml'):
        print(f"Il file {filename} non è un file SBML.")
        return

    print(f"\n=== Caricando file SBML: {filename} ===")
    
    # Carica il modello
    doc = libsbml.readSBML(file_path)
    if doc.getNumErrors() > 0:
        print(f"Attenzione: ci sono {doc.getNumErrors()} errori nel file SBML")
    
    model = doc.getModel()
    if model is None:
        print("Errore: modello SBML non trovato.")
        return
    
    print(f"\nModello: {model.getName()} (ID: {model.getId()})\n")
    
    '''# Compartimenti
    print("Compartimenti:")
    for comp in model.getListOfCompartments():
        print(f"  ID: {comp.getId()}, Nome: {comp.getName()}")
    print("\n")'''
    
    # Specie
    print("Specie:")
    for s in model.getListOfSpecies():
        print(f"  ID: {s.getId()}, Nome: {s.getName()}, Compartimento: {s.getCompartment()}")
    print("\n")
    
    '''# Reazioni
    print("Reazioni:")
    for reaction in model.getListOfReactions():
        print(f"Reazione: {reaction.getName()} (ID: {reaction.getId()})")
        print(f"  Compartimento: {reaction.getCompartment()}")
        
        reactants = [r.getSpecies() for r in reaction.getListOfReactants()]
        products = [p.getSpecies() for p in reaction.getListOfProducts()]
        
        print(f"  Reagenti: {reactants}")
        print(f"  Prodotti: {products}")
        print(f"  Reversibile: {reaction.getReversible()}")
        print("-----")'''

# Esempio di utilizzo:
leggi_sbml_file(os.path.join("working_homo-sapiens", "R-HSA-109582.sbml"))