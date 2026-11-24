# Esports War Feed

Bot che controlla YouTube ogni 15 minuti e pubblica in un canale del tuo server Discord
i video delle guerre di Clash of Clans giocate in **modalità Esports**: World Championship,
Monthly Qualifier/Final, LCQ, Town Hall Cup, scrim e tornei della community.

- **Gratis e sempre acceso**: gira su GitHub Actions, quindi non serve tenere acceso il PC.
- **Niente doppioni e niente buchi**: si ricorda i video già inoltrati e, se Discord non
  risponde, li tiene in coda per il giro successivo.
- **Dirette incluse**: le live programmate vengono inoltrate subito; il link diventa il VOD a fine diretta.
- **Forum opzionale**: se il canale è un Forum, ogni guerra diventa un post dove commentarla con la squadra.

Esempio di messaggio:

```
🎥 Tribe Gaming vs Repotted | World Championship Finals 2026
📺 Clash of Clans · 🗓️ 22 novembre 2026 15:00
🏷️ world championship
https://www.youtube.com/watch?v=...      ← Discord mostra il player del video
```

---

## Configurazione (circa 10 minuti)

### 1. Crea il webhook su Discord
1. Sul tuo server: **Impostazioni canale** (ingranaggio accanto al canale) → **Integrazioni** → **Webhook** → **Nuovo webhook**.
2. Scegli il canale dove vuoi ricevere i video e dagli un nome (es. *Esports War Feed*).
3. Premi **Copia URL webhook**. Trattalo come una password: chi ce l'ha può scrivere nel canale.

> Se il canale è di tipo **Forum**, in `config.toml` metti `forum_mode = true`.

### 2. Metti il bot su GitHub
1. Crea un account su [github.com](https://github.com) se non ce l'hai.
2. **New repository** → nome a piacere (es. `esports-war-feed`) → **Public** → *Create repository*.
   *Public* significa minuti illimitati gratis; l'URL del webhook resta comunque segreto.
3. Estrai lo zip, apri la cartella `esports-war-feed`, poi su GitHub **Add file → Upload files**
   e trascina **tutto quello che c'è dentro**, compresa la cartella `.github`.
   - Su Mac la cartella `.github` è nascosta: nel Finder premi `Cmd + Shift + .` per vederla.
   - In alternativa: **Add file → Create new file**, nome `.github/workflows/esports-war-feed.yml`,
     e incolla il contenuto di quel file.

### 3. Aggiungi il segreto
**Settings → Secrets and variables → Actions → New repository secret**
- Nome: `DISCORD_WEBHOOK_URL`
- Valore: l'URL copiato al passo 1

### 4. Prima prova
1. Scheda **Actions** → a sinistra **Esports War Feed** → **Run workflow**.
2. Spunta *Invia solo un messaggio di prova* → **Run workflow**: nel canale deve comparire un messaggio di conferma.
3. Rilancialo senza spunta: pubblica le guerre esports degli ultimi 14 giorni.

Da qui in poi parte da solo ogni 15 minuti.

---

## Facoltativo: ricerca su tutto YouTube

Senza chiave il bot legge solo i canali elencati in `config.toml`. Con una chiave gratuita
della YouTube Data API trova anche i video di canali che non conosci (ogni 3 ore, con 4 ricerche).

1. Vai su [console.cloud.google.com](https://console.cloud.google.com) e crea un progetto.
2. **API e servizi → Libreria** → cerca **YouTube Data API v3** → **Abilita**.
3. **Credenziali → Crea credenziali → Chiave API**. Conviene limitarla alla sola *YouTube Data API v3*.
4. Su GitHub aggiungi un secondo segreto: `YOUTUBE_API_KEY` = la chiave.

Il limite gratuito è di 10.000 unità al giorno; la configurazione di partenza ne usa circa 3.200.
Con la chiave il bot scarta anche i video sotto i 4 minuti (clip e Shorts), mentre le dirette restano.

---

## Personalizzare (`config.toml`)

Puoi modificarlo direttamente su GitHub (apri il file → icona matita → *Commit changes*).

| Cosa vuoi fare | Dove |
|---|---|
| Seguire un altro canale | aggiungi un blocco `[[channels]]` con `handle = "@nomecanale"` |
| Ricevere tutto da un canale (es. VOD della tua squadra) | nel suo blocco: `forward_all = true` |
| Seguire una playlist di un torneo | blocco `[[playlists]]` con l'id dopo `list=` nell'URL |
| Riconoscere un nuovo torneo | aggiungi il nome a `esports_keywords` |
| Togliere un tipo di video | aggiungi una parola a `exclude_in_title` |
| Avvisare un ruolo | `mention = "<@&ID_DEL_RUOLO>"` (tasto destro sul ruolo → Copia ID, con la modalità sviluppatore attiva) |
| Recuperare più storico | alza `max_age_days` |

Canali già inclusi: Clash of Clans ufficiale (dove vanno in onda Worlds, Monthly Final e Town Hall Cup),
Blueprint CoC, CarbonFin Gaming e Judo Sloth Gaming.

---

## In alternativa: sul tuo PC

Serve Python 3.11 o successivo, senza librerie da installare.

```bash
cp .env.example .env        # su Windows: copy .env.example .env
# apri .env e incolla l'URL del webhook
python bot.py --test        # messaggio di prova
python bot.py --dry-run -v  # mostra cosa pubblicherebbe e perché scarta il resto
python bot.py --loop 15     # resta acceso e controlla ogni 15 minuti
```

Funziona solo finché il PC è acceso e connesso a internet; per questo GitHub è la scelta consigliata.

---

## Limiti da sapere

- **La modalità Esports non si vede dai dati di YouTube.** Il bot la riconosce da titolo, descrizione,
  tag e canale. I tornei ufficiali (Worlds, Monthly, LCQ, Town Hall Cup) si giocano tutti in modalità Esports,
  quindi sono coperti; per tornei della community con nomi particolari aggiungi il nome a `esports_keywords`.
- I feed di YouTube mostrano gli ultimi 15 video per canale: con un controllo ogni 15 minuti non è un problema.
- GitHub a volte avvia i controlli programmati con qualche minuto di ritardo.
- Le VOD solo su Twitch non sono coperte.
- Se un repository pubblico resta 60 giorni senza attività, GitHub sospende i controlli programmati.
  Il bot aggiorna `state.json` almeno una volta a settimana per evitarlo; se succede comunque,
  basta riattivarlo dalla scheda **Actions**.

## Problemi comuni

| Messaggio nel log (Actions → ultimo run) | Soluzione |
|---|---|
| `Manca DISCORD_WEBHOOK_URL` | controlla il nome del segreto, deve essere identico |
| `Discord HTTP 404` | il webhook è stato eliminato: creane uno nuovo e aggiorna il segreto |
| `il canale è un Forum` | metti `forum_mode = true` |
| `quota giornaliera esaurita` | riduci `queries` o alza `every_hours` |
| `Canale '...': non trovo l'ID` | controlla l'handle oppure metti direttamente `id = "UC..."` |
| errore `git push` / permessi | Settings → Actions → General → Workflow permissions → *Read and write* |

Per rivedere tutto da capo cancella `state.json` dal repository: al giro dopo ripubblica le guerre degli ultimi `max_age_days` giorni.
