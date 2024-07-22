import random


def scegli_parola():
    parole = ["gatto", "cane", "computer", "programmazione", "python", "impiccato", "gioco"]
    return random.choice(parole)


def gioca_impiccato():
    parola = scegli_parola()
    lunghezza_parola = len(parola)
    lettere_indovinate = ['_'] * lunghezza_parola
    tentativi_rimasti = 6
    lettere_usate = []

    print("Benvenuto al gioco dell'impiccato!")
    print("Indovina la parola indovinando una lettera alla volta.")
    print(f"La parola è lunga {lunghezza_parola} lettere.")

    while tentativi_rimasti > 0:
        print("\nParola da indovinare:", " ".join(lettere_indovinate))
        print(f"Tentativi rimasti: {tentativi_rimasti}")
        lettera = input("Indovina una lettera: ").lower()

        if len(lettera) != 1 or not lettera.isalpha():
            print("Per favore, inserisci una singola lettera valida.")
            continue

        if lettera in lettere_usate:
            print("Hai già usato questa lettera. Prova con un'altra.")
            continue

        lettere_usate.append(lettera)

        if lettera in parola:
            for i in range(lunghezza_parola):
                if parola[i] == lettera:
                    lettere_indovinate[i] = lettera
            if '_' not in lettere_indovinate:
                print("\nComplimenti! Hai indovinato la parola:", parola)
                break
        else:
            tentativi_rimasti -= 1
            print("La lettera", lettera, "non è presente nella parola.")

    if '_' in lettere_indovinate:
        print("\nMi dispiace, hai esaurito i tentativi. La parola era:", parola)


if __name__ == "__main__":
    gioca_impiccato()

