def calcola_area_cerchio(raggio):
    area = 3.14 * raggio ** 2
    return area

def calcola_media(lista_numeri):
    if not lista_numeri:
        return 0
    return sum(lista_numeri) / len(lista_numeri)

def main():
    print("Benvenuto nel programma Python di esempio!")
    nome = input("Qual è il tuo nome? ")
    print(f"Ciao, {nome}! In questo programma ti mostreremo alcuni concetti base di Python.")

    # Esempio di variabile e input dell'utente
    print("\nEsempio di variabile e input dell'utente:")
    eta = int(input("Quanti anni hai? "))
    anno_nascita = 2024 - eta
    print(f"Se hai {eta} anni, sei nato intorno al {anno_nascita}.")

    # Esempio di condizionale
    print("\nEsempio di condizionale (if-else):")
    if eta >= 18:
        print("Sei maggiorenne.")
    else:
        print("Sei minorenne.")

    # Esempio di ciclo
    print("\nEsempio di ciclo (for loop):")
    numeri = [1, 2, 3, 4, 5]
    for numero in numeri:
        quadrato = numero ** 2
        print(f"Il quadrato di {numero} è {quadrato}.")

    # Esempio di funzione
    print("\nEsempio di funzione:")
    voto1 = float(input("Inserisci il primo voto: "))
    voto2 = float(input("Inserisci il secondo voto: "))
    voto3 = float(input("Inserisci il terzo voto: "))
    voti = [voto1, voto2, voto3]
    media = calcola_media(voti)
    print(f"La media dei voti è: {media:.2f}")

    # Esempio di utilizzo di una funzione per calcolare l'area di un cerchio
    print("\nEsempio di funzione per calcolare l'area di un cerchio:")
    raggio_cerchio = float(input("Inserisci il raggio del cerchio: "))
    area_cerchio = calcola_area_cerchio(raggio_cerchio)
    print(f"L'area del cerchio con raggio {raggio_cerchio} è: {area_cerchio:.2f}")

    # Saluto finale
    print("\nGrazie per aver utilizzato il programma di esempio!")
    print("Spero che ti sia stato utile per capire alcuni concetti di base di Python.")

if __name__ == "__main__":
    main()
