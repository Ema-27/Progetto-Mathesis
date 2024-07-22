import statistics


class Studente:
    def __init__(self, nome, cognome):
        self.nome = nome
        self.cognome = cognome
        self.voti = []

    def aggiungi_voto(self, voto):
        self.voti.append(voto)

    def media_voti(self):
        if len(self.voti) == 0:
            return 0
        return sum(self.voti) / len(self.voti)

    def mediana_voti(self):
        if len(self.voti) == 0:
            return 0
        return statistics.median(self.voti)

    def distribuzione_voti(self):
        sopra_sei = 0
        sotto_sei = 0
        for voto in self.voti:
            if voto >= 6:
                sopra_sei += 1
            else:
                sotto_sei += 1
        totale = len(self.voti)
        if totale == 0:
            return 0, 0
        percentuale_sopra_sei = (sopra_sei / totale) * 100
        percentuale_sotto_sei = (sotto_sei / totale) * 100
        return percentuale_sopra_sei, percentuale_sotto_sei

    def risultato(self):
        media = self.media_voti()
        if media >= 6:
            return "Superato"
        else:
            return "Non Superato"

    def __str__(self):
        return f"{self.cognome} {self.nome}"


def salva_dati(studenti, nome_file):
    with open(nome_file, 'w') as file:
        for studente in studenti:
            voti_str = ' '.join(map(str, studente.voti))
            file.write(f"{studente.cognome},{studente.nome}:{voti_str}\n")


def carica_dati(nome_file):
    studenti = []
    try:
        with open(nome_file, 'r') as file:
            for line in file:
                info, voti_str = line.strip().split(':')
                cognome, nome = info.split(',')
                voti = voti_str.split()
                voti = [float(voto) for voto in voti] if voti else []
                studente = Studente(nome, cognome)
                studente.voti = voti
                studenti.append(studente)
    except FileNotFoundError:
        pass
    return studenti


def trova_studente(studenti, nome, cognome):
    for studente in studenti:
        if studente.nome == nome and studente.cognome == cognome:
            return studente
    return None


def main():
    nome_file = 'studenti.txt'
    studenti = carica_dati(nome_file)
    ultimo_studente = None

    while True:
        print("\n1. Aggiungi Studente")
        print("2. Aggiungi Voto a Studente")
        print("3. Mostra Media e Risultato di uno Studente")
        print("4. Mostra Mediana dei Voti di uno Studente")
        print("5. Mostra Distribuzione dei Voti di uno Studente")
        print("6. Salva i dati")
        print("7. Esci")

        scelta = input("Scegli un'opzione: ")

        if scelta == '1':
            nome = input("Inserisci il nome dello studente: ")
            cognome = input("Inserisci il cognome dello studente: ")
            nuovo_studente = Studente(nome, cognome)
            studenti.append(nuovo_studente)
            ultimo_studente = nuovo_studente

        elif scelta == '2':
            cambia_studente = input("Vuoi selezionare un altro studente? (sì/no): ").strip().lower()
            if cambia_studente == 'sì' or cambia_studente == 'si':
                nome = input("Inserisci il nome dello studente: ")
                cognome = input("Inserisci il cognome dello studente: ")
                studente = trova_studente(studenti, nome, cognome)
            else:
                studente = ultimo_studente

            if studente is not None:
                voto = float(input(f"Inserisci il voto per {studente}: "))
                studente.aggiungi_voto(voto)
                ultimo_studente = studente
            else:
                print("Studente non trovato.")
                ultimo_studente = None

        elif scelta == '3':
            if ultimo_studente is None:
                nome = input("Inserisci il nome dello studente: ")
                cognome = input("Inserisci il cognome dello studente: ")
                studente = trova_studente(studenti, nome, cognome)
            else:
                studente = ultimo_studente

            if studente is not None:
                print(f"Media voti di {studente}: {studente.media_voti():.2f}")
                print(f"Risultato: {studente.risultato()}")
                ultimo_studente = studente
            else:
                print("Studente non trovato.")
                ultimo_studente = None

        elif scelta == '4':
            if ultimo_studente is None:
                nome = input("Inserisci il nome dello studente: ")
                cognome = input("Inserisci il cognome dello studente: ")
                studente = trova_studente(studenti, nome, cognome)
            else:
                studente = ultimo_studente

            if studente is not None:
                print(f"Mediana voti di {studente}: {studente.mediana_voti():.2f}")
                ultimo_studente = studente
            else:
                print("Studente non trovato.")
                ultimo_studente = None

        elif scelta == '5':
            if ultimo_studente is None:
                nome = input("Inserisci il nome dello studente: ")
                cognome = input("Inserisci il cognome dello studente: ")
                studente = trova_studente(studenti, nome, cognome)
            else:
                studente = ultimo_studente

            if studente is not None:
                percentuale_sopra_sei, percentuale_sotto_sei = studente.distribuzione_voti()
                print(f"Percentuale voti >= 6: {percentuale_sopra_sei:.2f}%")
                print(f"Percentuale voti < 6: {percentuale_sotto_sei:.2f}%")
                ultimo_studente = studente
            else:
                print("Studente non trovato.")
                ultimo_studente = None

        elif scelta == '6':
            salva_dati(studenti, nome_file)
            print("Dati salvati correttamente.")

        elif scelta == '7':
            salva_dati(studenti, nome_file)
            print("Uscita dal programma e dati salvati.")
            break

        else:
            print("Opzione non valida. Riprova.")


if __name__ == "__main__":
    main()
