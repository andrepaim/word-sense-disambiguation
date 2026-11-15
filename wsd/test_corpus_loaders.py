"""Training-corpus loaders: Raganato-format XML (SemCor, SemEval ALL) and the WordNet Gloss Corpus."""
from training.semcor import detokenize, load_raganato, load_sense_index
from training.wngt import load_wngt

# WordNet 3.0 index.sense lines: sense key, synset offset, sense number, tag count.
INDEX_SENSE = """\
bowl%1:06:00:: 02881193 1 3
dog%1:05:00:: 02084071 1 42
dog%1:05:01:: 02085998 5 0
domesticated%3:00:00:tame:01 01133876 1 2
genus_canis%1:05:00:: 02083863 1 0
man%1:05:01:: 02472293 2 30
prehistoric%3:00:00:past:00 01730444 1 1
run_away%2:38:00:: 02073233 1 2
"""

RAGANATO_XML = """\
<?xml version="1.0" encoding="UTF-8" ?>
<corpus lang="en" source="test">
<text id="d000">
<sentence id="d000.s000">
<wf lemma="the" pos="DET">The</wf>
<instance id="d000.s000.t000" lemma="dog" pos="NOUN">dog</instance>
<wf lemma="'s" pos="PRT">'s</wf>
<instance id="d000.s000.t001" lemma="bowl" pos="NOUN">bowl</instance>
<wf lemma="," pos=".">,</wf>
<wf lemma="and" pos="CONJ">and</wf>
<wf lemma="the" pos="DET">the</wf>
<instance id="d000.s000.t002" lemma="dog" pos="NOUN">dog</instance>
<instance id="d000.s000.t003" lemma="quickly" pos="ADV">quickly</instance>
<instance id="d000.s000.t004" lemma="run_away" pos="VERB">ran away</instance>
<wf lemma="." pos=".">.</wf>
</sentence>
<sentence id="d000.s001">
<wf lemma="5*3" pos="NUM">5*3</wf>
<instance id="d000.s001.t000" lemma="dog" pos="NOUN">dog</instance>
</sentence>
</text>
</corpus>
"""

RAGANATO_GOLD = """\
d000.s000.t000 dog%1:05:00:: dog%1:05:01::
d000.s000.t001 bowl%1:06:00::
d000.s000.t002 dog%1:05:00::
d000.s000.t003 quickly%4:02:00::
d000.s000.t004 run_away%2:38:00::
d000.s001.t000 dog%1:05:00::
"""

# The real WordNet-3.0-glosstag entry for "dog" (merged/noun.xml), trimmed: ignored words, a collocation
# (<cf>/<glob>), an <aux> parenthetical, manually tagged words and the separator token that ends a gloss.
GLOSSTAG_NOUN_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<wordnet ver="3.0">
 <synset id="n02084071" ofs="02084071" pos="n">
  <gloss desc="wsd">
   <def id="n02084071_d">
    <wf id="n02084071_wf1" lemma="a" pos="DT" tag="ignore">a</wf>
    <wf id="n02084071_wf2" lemma="member%1" pos="NN" tag="un">member</wf>
    <wf id="n02084071_wf3" lemma="of" pos="IN" tag="ignore">of</wf>
    <wf id="n02084071_wf4" lemma="the" pos="DT" tag="ignore">the</wf>
    <cf coll="a" id="n02084071_wf5" lemma="genus%1" pos="NN" tag="un">
     <glob coll="a" glob="auto" id="n02084071_coll.a" lemma="genus_Canis%1" tag="auto">
      <id coll="a" id="n02084071_id.2" lemma="genus Canis" sk="genus_canis%1:05:00::"/>
    </glob>genus</cf>
    <cf coll="a" id="n02084071_wf6" lemma="Canis%1" pos="NNP" tag="un">Canis</cf>
    <aux tag="ignore">
     <wf id="n02084071_tok1" pos="(" sep="" tag="ignore" type="punc">(</wf>
     <wf id="n02084071_tok2" lemma="probably%4" pos="RB" tag="un">probably</wf>
     <wf id="n02084071_tok8" pos=")" tag="ignore" type="punc">)</wf>
    </aux>
    <wf id="n02084071_wf7" lemma="that" pos="WDT" tag="ignore">that</wf>
    <wf id="n02084071_wf8" lemma="have%2" pos="VBZ" tag="un">has</wf>
    <wf id="n02084071_wf9" lemma="be%2" pos="VBN" tag="un">been</wf>
    <wf id="n02084071_wf10" lemma="domesticate%2|domesticated%3" pos="VBN" tag="man">
     <id id="n02084071_id.5" lemma="domesticated" sk="domesticated%3:00:00:tame:01"/>domesticated</wf>
    <wf id="n02084071_wf11" lemma="by" pos="IN" tag="ignore">by</wf>
    <wf id="n02084071_wf12" lemma="man%1|man%2" pos="NN" tag="man">
     <id id="n02084071_id.3" lemma="man" sk="man%1:05:01::"/>man</wf>
    <wf id="n02084071_wf13" lemma="since" pos="IN" tag="ignore">since</wf>
    <wf id="n02084071_wf14" lemma="prehistoric%3" pos="JJ" tag="man">
     <id id="n02084071_id.4" lemma="prehistoric" sk="prehistoric%3:00:00:past:00"/>prehistoric</wf>
    <wf id="n02084071_wf15" lemma="time%1|times%1|time%2" pos="NNS" sep="" tag="un">times</wf>
    <wf id="n02084071_wf21" pos=":" tag="ignore" type="punc">;</wf>
   </def>
  </gloss>
 </synset>
</wordnet>
"""

EMPTY_GLOSSTAG = '<?xml version="1.0" encoding="UTF-8"?>\n<wordnet ver="3.0"/>\n'


def _sense_index(tmp_path):
    path = tmp_path / "index.sense"
    path.write_text(INDEX_SENSE)
    return load_sense_index(path)


def test_detokenize_ptb_tokens():
    tokens = "the dog 's bowl , and ( maybe ) `` more '' .".split()
    assert detokenize(tokens) == 'the dog\'s bowl, and (maybe) "more".'


def test_sense_index_maps_keys_to_omw_ids(tmp_path):
    index = _sense_index(tmp_path)
    assert index["dog%1:05:00::"] == "omw-en-02084071-n"
    assert index["run_away%2:38:00::"] == "omw-en-02073233-v"
    assert index["domesticated%3:00:00:tame:01"] == "omw-en-01133876-a"


def test_raganato_marks_each_instance_in_detokenized_text(tmp_path):
    (tmp_path / "corpus.data.xml").write_text(RAGANATO_XML)
    (tmp_path / "corpus.gold.key.txt").write_text(RAGANATO_GOLD)
    examples = load_raganato(tmp_path / "corpus", _sense_index(tmp_path))

    # quickly: its sense key is not in the index; the second sentence contains the marker character.
    assert [(e.word_form, e.marked_text) for e in examples] == [
        ("dog", "The *dog*'s bowl, and the dog quickly ran away."),
        ("bowl", "The dog's *bowl*, and the dog quickly ran away."),
        ("dog", "The dog's bowl, and the *dog* quickly ran away."),  # the repeated word, at its own position
        ("ran away", "The dog's bowl, and the dog quickly *ran away*."),
    ]
    first, _, _, phrasal = examples
    assert first.sentence == "The dog's bowl, and the dog quickly ran away."
    assert (first.synset_id, first.gold_ids, first.pos) == ("omw-en-02084071-n", ("omw-en-02085998-n",), "n")
    assert (phrasal.lemma, phrasal.pos, phrasal.synset_id) == ("run away", "v", "omw-en-02073233-v")


def _glosstag_dir(tmp_path):
    merged = tmp_path / "glosstag" / "merged"
    merged.mkdir(parents=True)
    (merged / "noun.xml").write_text(GLOSSTAG_NOUN_XML)
    for name in ("verb.xml", "adj.xml", "adv.xml"):
        (merged / name).write_text(EMPTY_GLOSSTAG)
    return tmp_path / "glosstag"


def test_wngt_manual_tags_of_definitions(tmp_path):
    examples = load_wngt(_glosstag_dir(tmp_path), _sense_index(tmp_path), parts=frozenset({"def"}),
                         tags=frozenset({"man"}))

    assert [(e.lemma, e.pos, e.synset_id) for e in examples] == [
        ("domesticated", "a", "omw-en-01133876-a"),
        ("man", "n", "omw-en-02472293-n"),
        ("prehistoric", "a", "omw-en-01730444-a"),
    ]
    for e in examples:
        assert e.marked_text.count("*") == 2
        assert f"*{e.word_form}*" in e.marked_text
        assert not e.sentence.endswith(";")  # the gloss-ending separator token is dropped


def test_wngt_collocation_spans_its_tokens(tmp_path):
    examples = load_wngt(_glosstag_dir(tmp_path), _sense_index(tmp_path), parts=frozenset({"def"}),
                         tags=frozenset({"auto"}))

    assert len(examples) == 1
    (collocation,) = examples
    assert (collocation.word_form, collocation.lemma, collocation.synset_id) == (
        "genus Canis", "genus Canis", "omw-en-02083863-n",
    )
    assert "*genus Canis*" in collocation.marked_text


def test_wngt_skips_unselected_parts(tmp_path):
    assert load_wngt(_glosstag_dir(tmp_path), _sense_index(tmp_path), parts=frozenset({"ex"})) == []
