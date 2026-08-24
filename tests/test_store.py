from datetime import datetime, timedelta, timezone

from telegram_rag.store import Store


def _add(store, embedder, msg_id, text, *, date=None, chat_id=1):
    [vec] = embedder.embed([text])
    store.add(
        chat_id=chat_id,
        msg_id=msg_id,
        date=date or datetime.now(timezone.utc),
        sender="tester",
        text=text,
        embedding=vec,
    )


def test_add_and_count(store, embedder):
    assert store.count() == 0
    _add(store, embedder, 1, "BBVA bonificación restaurantes")
    _add(store, embedder, 2, "Banorte 12 MSI en Amazon")
    assert store.count() == 2


def test_duplicate_message_ignored(store, embedder):
    _add(store, embedder, 1, "BBVA bonificación restaurantes")
    _add(store, embedder, 1, "BBVA bonificación restaurantes")
    assert store.count() == 1


def test_same_msg_id_different_chat_kept(store, embedder):
    _add(store, embedder, 1, "BBVA promo", chat_id=1)
    _add(store, embedder, 1, "Banorte promo", chat_id=2)
    assert store.count() == 2


def test_search_ranks_by_similarity(store, embedder):
    _add(store, embedder, 1, "BBVA tiene 20% de bonificación en restaurantes")
    _add(store, embedder, 2, "Banorte lanza 12 MSI en Amazon")
    _add(store, embedder, 3, "El clima está horrible hoy")

    [qvec] = embedder.embed(["promociones BBVA tarjeta"])
    hits = store.search(qvec, limit=3)
    assert hits[0].text.startswith("BBVA")
    assert hits[0].score > hits[-1].score


def test_search_limit(store, embedder):
    for i in range(5):
        _add(store, embedder, i, f"BBVA promo número {i}")
    [qvec] = embedder.embed(["bbva"])
    assert len(store.search(qvec, limit=2)) == 2


def test_search_since_filter(store, embedder):
    old = datetime.now(timezone.utc) - timedelta(days=365)
    _add(store, embedder, 1, "BBVA promo vieja", date=old)
    _add(store, embedder, 2, "BBVA promo nueva")

    [qvec] = embedder.embed(["bbva"])
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d %H:%M")
    hits = store.search(qvec, limit=10, since=cutoff)
    assert [h.text for h in hits] == ["BBVA promo nueva"]


def test_search_empty_store(store, embedder):
    [qvec] = embedder.embed(["bbva"])
    assert store.search(qvec, limit=5) == []


def test_persistence_across_reopen(tmp_path, embedder):
    path = str(tmp_path / "p.db")
    first = Store(path)
    [vec] = embedder.embed(["BBVA promo"])
    first.add(
        chat_id=1, msg_id=1, date=datetime.now(timezone.utc),
        sender="t", text="BBVA promo", embedding=vec,
    )

    reopened = Store(path)
    assert reopened.count() == 1
    assert 1 in reopened.known_ids(1)
    [qvec] = embedder.embed(["bbva"])
    assert reopened.search(qvec, limit=1)[0].text == "BBVA promo"


def test_latest_date(store, embedder):
    assert store.latest_date() is None
    _add(store, embedder, 1, "BBVA promo", date=datetime(2026, 1, 1, tzinfo=timezone.utc))
    _add(store, embedder, 2, "Banorte promo", date=datetime(2026, 6, 1, tzinfo=timezone.utc))
    assert store.latest_date().startswith("2026-06-01")


def test_hybrid_acronym_beats_semantic_noise(store, embedder):
    # An acronym query gives cosine no signal (no keyword dims in the fake
    # embedder), which mirrors reality: dense similarity between "SV" and
    # short chat messages is noise. The keyword bonus must decide.
    _add(store, embedder, 1, "El clima está horrible hoy")
    _add(store, embedder, 2, "Noupe, con SV no pasa nada")

    [qvec] = embedder.embed(["SV significado"])
    hits = store.search(qvec, limit=2, query_text="SV significado")
    assert "SV" in hits[0].text


def test_hybrid_short_token_respects_word_boundary(store, embedder):
    _add(store, embedder, 1, "misvacaciones fueron caras")  # 'sv' inside a word
    _add(store, embedder, 2, "con SV no aplica la promo")

    [qvec] = embedder.embed(["algo sin señal"])
    hits = store.search(qvec, limit=2, query_text="sv")
    assert "SV" in hits[0].text


def test_hybrid_no_terms_falls_back_to_cosine(store, embedder):
    _add(store, embedder, 1, "BBVA promo restaurantes")
    _add(store, embedder, 2, "Banorte MSI Amazon")
    [qvec] = embedder.embed(["bbva"])
    # stopword-only query text contributes no bonus
    hits = store.search(qvec, limit=1, query_text="que como para")
    assert hits[0].text.startswith("BBVA")
