import os, json, random, gc
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

MODELS = os.environ.get("ATLAS_MODELS", "Qwen/Qwen2.5-1.5B,bigscience/bloom-1b1,Qwen/Qwen2.5-0.5B").split(",")
MIN_GAP = float(os.environ.get("ATLAS_GAP", "1.0"))
dev = "cuda" if torch.cuda.is_available() else "cpu"

FEM = ("casa mesa silla puerta ventana playa cama ropa carta falda plaza escuela luna taza botella sopa semana palabra historia "
       "familia cocina tienda iglesia empresa bolsa caja hoja rosa pluma tierra vida isla lluvia sombra camisa moneda carretera "
       "montaña pregunta respuesta idea forma cosa persona música lengua fiesta cena comida boca cara cabeza espalda "
       "ciudad flor luz noche canción razón nube llave leche piel voz paz red ley calle carne mujer nariz verdad sangre sal clase "
       "parte gente muerte frase madre imagen crisis tesis serie especie fuente torre nave base mente suerte libertad universidad "
       "edad pared cruz vez nuez mano foto moto").split()
MASC = ("libro perro gato camino vaso cielo mundo tiempo trabajo juego puerto techo suelo pelo barco pueblo muro banco cuaderno "
        "zapato brazo dedo amigo vecino cuerpo ojo oído dinero trozo hueso vuelo viento fuego hielo campo lado grupo cuento ejemplo "
        "edificio coche río sol tren árbol pan mar lápiz país color amor papel motor jardín avión reloj lugar nombre puente parque "
        "viaje balcón camión corazón dolor valor sabor calor olor mes pie diente aire café té hombre padre rey pez "
        "día mapa planeta problema sistema idioma tema clima programa poema teorema esquema lema diploma drama dilema síntoma "
        "fantasma aroma pijama tranvía").split()
ADJ = [("blanca","blanco"),("roja","rojo"),("nueva","nuevo"),("vieja","viejo"),("bonita","bonito"),("pequeña","pequeño"),
       ("alta","alto"),("baja","bajo"),("larga","largo"),("buena","bueno"),("mala","malo"),("negra","negro"),("limpia","limpio"),
       ("sucia","sucio"),("cerrada","cerrado"),("abierta","abierto"),("famosa","famoso"),("hermosa","hermoso"),("lenta","lento"),
       ("rápida","rápido"),("fría","frío")]

def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 3), round(float(c + h), 3)]

def grp(n, g):
    if (g == "f" and n.endswith("a")) or (g == "m" and n.endswith("o")): return "regular"
    if (g == "f" and n.endswith("o")) or (g == "m" and n.endswith("a")): return "trampa"
    return "sin sufijo"

def attn_out(layer):
    for an in ("self_attn", "self_attention", "attn", "attention"):
        a = getattr(layer, an, None)
        if a is not None:
            for pn in ("o_proj", "dense", "out_proj", "c_proj"):
                p = getattr(a, pn, None)
                if p is not None: return p
    raise RuntimeError("No encuentro la proyección de salida de la atención")

def run_model(MODEL):
    tok = AutoTokenizer.from_pretrained(os.environ.get("ATLAS_TOK") or MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).float().to(dev).eval()
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    one = lambda w: len(tok.encode(" " + w, add_special_tokens=False)) == 1
    adj = [(f, m) for f, m in ADJ if one(f) and one(m)]
    assert len(adj) >= 2, "Casi ningún adjetivo cabe en un token"
    fem_ids = torch.tensor([tok.encode(" " + f, add_special_tokens=False)[0] for f, _ in adj], device=dev)
    masc_ids = torch.tensor([tok.encode(" " + m, add_special_tokens=False)[0] for _, m in adj], device=dev)
    tl = lambda a, n: len(tok(f"{a} {n} es").input_ids)
    T = min(tl("Mi", n) for n in set(FEM + MASC))
    keep = lambda n: all(tl(a, n) == T for a in ("Mi", "El", "La"))
    words = [(n, "f") for n in FEM if keep(n)] + [(n, "m") for n in MASC if keep(n)]
    ws = [w for w, _ in words]; gs = [g for _, g in words]; grps = [grp(w, g) for w, g in words]
    y = np.array([g == "f" for g in gs]); yy = y.astype(int)
    print(f"[{MODEL}] {len(ws)} sustantivos, {len(adj)} pares de adjetivos")
    layers = None; n_l = model.config.num_hidden_layers
    for _, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == n_l: layers = mod; break
    outs = [attn_out(l) for l in layers]; nl = len(layers); H = model.config.num_attention_heads; dh = outs[0].in_features // H
    encp = lambda ps: tok(ps, return_tensors="pt").input_ids.to(dev)
    enc = lambda xs: encp([f"Mi {w} es" for w in xs])
    ld = lambda lg: (lg[:, fem_ids] - lg[:, masc_ids]).mean(1)

    @torch.no_grad()
    def forward(ids, patch=None, hi=None, src=None, rec=False):
        store, hst, hs = [], [], []
        for l, layer in enumerate(layers):
            def post(mod, inp, out, l=l):
                h = out[0] if isinstance(out, tuple) else out
                store.append(h.detach())
                if patch and patch[0] == l:
                    h = h.clone(); h[:, patch[1]] = patch[2]
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            def pre(mod, args, l=l):
                x = args[0]
                if rec: hst.append(x[:, -1].detach().clone())
                if hi and l in hi:
                    x = x.clone()
                    for h_ in hi[l]:
                        a, b = h_ * dh, (h_ + 1) * dh; x[:, -1, a:b] = src[l][..., a:b]
                    return (x,) + tuple(args[1:])
            hs.append(layer.register_forward_hook(post)); hs.append(outs[l].register_forward_pre_hook(pre))
        try: lg = model(ids).logits[:, -1].float()
        finally:
            for h in hs: h.remove()
        return lg, store, hst

    # 1) Conducta
    lg, res, _ = forward(enc(ws)); L0 = ld(lg).cpu().numpy(); okb = ((L0 > 0) == y); beh = {}
    for k in sorted(set(grps)):
        idx = [i for i, x in enumerate(grps) if x == k]; c = int(okb[idx].sum()); beh[k] = dict(acc=round(c / len(idx), 3), n=len(idx), ci=wilson(c, len(idx)))
    c = int(okb.sum()); beh["todo"] = dict(acc=round(c / len(ws), 3), n=len(ws), ci=wilson(c, len(ws)))
    margins = dict(f=round(float(L0[y].mean()), 2), m=round(float(-L0[~y].mean()), 2))
    nouns = [dict(w=w, g=g, grp=r, ld=round(float(v), 2)) for w, g, r, v in zip(ws, gs, grps, L0)]
    # 1b) Conflicto de pistas: artículo contra sustantivo
    art = lambda g, c: "La" if (g == "f") == c else "El"
    dcg = ld(forward(encp([f"{art(g, True)} {w} es" for w, g in zip(ws, gs)]))[0]).cpu().numpy()
    dig = ld(forward(encp([f"{art(g, False)} {w} es" for w, g in zip(ws, gs)]))[0]).cpu().numpy()
    fol = ((dig > 0) == y)
    cue = dict(congruente=round(float(((dcg > 0) == y).mean()), 3), sigue_sustantivo=round(float(fol.mean()), 3),
               por_grupo={k: round(float(fol[[i for i, x in enumerate(grps) if x == k]].mean()), 3) for k in sorted(set(grps))})
    print("  acierto", {k: v["acc"] for k, v in beh.items()}, "| conflicto: sigue al sustantivo", cue["sigue_sustantivo"])

    # 2) Parcheo por capa/posición y por cabeza (en la última palabra)
    pos = [T - 2, T - 1]; rg = random.Random(0)
    F = [w for w, g in words if g == "f"]; M = [w for w, g in words if g == "m"]; rg.shuffle(F); rg.shuffle(M); n = min(len(F), len(M)); F, M = F[:n], M[:n]
    recs, hrec, hcnt = [], np.zeros((nl, H)), 0
    for cw, kw, cfem in [(F, M, True), (M, F, False)]:
        ic, ik = enc(cw), enc(kw); lc, rc, hc = forward(ic, rec=True); lk, _, _ = forward(ik); dc, dk = ld(lc), ld(lk)
        good = ((dc > 0) == cfem) & ((dk > 0) != cfem) & ((dc - dk).abs() > MIN_GAP); ng = int(good.sum())
        if ng == 0: continue
        R = np.zeros((nl, 2, ng))
        for l in range(nl):
            for pi, p in enumerate(pos):
                lp, _, _ = forward(ik, patch=(l, p, rc[l][:, p])); R[l, pi] = ((ld(lp) - dk) / (dc - dk))[good].cpu().numpy()
            for h in range(H):
                lp, _, _ = forward(ik, hi={l: [h]}, src=hc); hrec[l, h] += ((ld(lp) - dk) / (dc - dk))[good].sum().item()
        recs.append(R); hcnt += ng
    if recs:
        Ra = np.nan_to_num(np.concatenate(recs, 2)); N = Ra.shape[2]; bi = np.random.RandomState(0).randint(0, N, (400, N))
        bm = Ra[:, :, bi].mean(-1)
        patch = dict(mean=Ra.mean(2).round(3).tolist(), lo=np.percentile(bm, 2.5, axis=2).round(3).tolist(), hi=np.percentile(bm, 97.5, axis=2).round(3).tolist())
        hm = np.nan_to_num(hrec / hcnt); top = sorted([(float(hm[l, h]), l, h) for l in range(nl) for h in range(H)], reverse=True)[:12]
        heads = dict(grid=hm.round(3).tolist(), top=[[l, h, round(v, 3)] for v, l, h in top])
    else: N, patch, heads = 0, None, None

    # 3) Sondas: entrenadas con regulares, probadas con regulares nuevos (validación cruzada), trampas y sin sufijo
    reg = np.array([i for i, r in enumerate(grps) if r == "regular"]); trap = [i for i, r in enumerate(grps) if r == "trampa"]; neu = [i for i, r in enumerate(grps) if r == "sin sufijo"]
    nspl = min(5, int(np.bincount(yy[reg]).min())); probe = {"regular": [], "trampa": [], "sin sufijo": []}
    for l in range(nl):
        X = res[l][:, T - 2].float().cpu().numpy(); fit = lambda a: (StandardScaler().fit(X[a]), a)
        accs = []
        if nspl >= 2:
            for tr, te in StratifiedKFold(nspl, shuffle=True, random_state=0).split(reg, yy[reg]):
                sc = StandardScaler().fit(X[reg[tr]]); clf = LogisticRegression(max_iter=3000, C=0.05).fit(sc.transform(X[reg[tr]]), yy[reg[tr]])
                accs.append((clf.predict(sc.transform(X[reg[te]])) == yy[reg[te]]).mean())
        sc = StandardScaler().fit(X[reg]); clf = LogisticRegression(max_iter=3000, C=0.05).fit(sc.transform(X[reg]), yy[reg])
        probe["regular"].append(round(float(np.mean(accs)), 3) if accs else None)
        for k_, idx in (("trampa", trap), ("sin sufijo", neu)):
            probe[k_].append(round(float((clf.predict(sc.transform(X[idx])) == yy[idx]).mean()), 3) if idx else None)
    # 4) Necesidad y especificidad: apagar (con la media) las cabezas clave frente a cabezas al azar
    ablation = None
    if heads:
        best = heads["top"][0][2]
        K = max(2, min(6, sum(1 for _, _, v in heads["top"] if v >= max(0.05, 0.25 * best))))
        topK = [(l, h) for l, h, _ in heads["top"][:K]]
        mk = lambda hs: {l_: [h_ for l2, h_ in hs if l2 == l_] for l_ in {l2 for l2, _ in hs}}
        mu = [x.mean(0) for x in forward(enc(ws), rec=True)[2]]
        def metrics(prompts, yv, hs=None):
            d = ld(forward(encp(prompts), hi=mk(hs) if hs else None, src=mu)[0]).cpu().numpy(); sm = np.where(yv, d, -d)
            return [round(float((sm > 0).mean()), 3), round(float(sm.mean()), 2)]
        rg2 = random.Random(1); Fw = [w for w, g in words if g == "f"]; Mw = [w for w, g in words if g == "m"]
        pf = [(rg2.choice(Fw), rg2.choice(Mw)) for _ in range(120)]
        tsets = {"directa": ([f"Mi {w} es" for w in ws], y),
                 "distractor masculino": ([f"Mi {a} del {b} es" for a, b in pf], np.ones(len(pf), bool)),
                 "distractor femenino": ([f"Mi {b} de la {a} es" for a, b in pf], np.zeros(len(pf), bool))}
        allh = [(l, h) for l in range(nl) for h in range(H)]; rr = random.Random(2)
        ablation = dict(K=K, heads=[[l, h] for l, h in topK], sets={})
        for name, (ps, yv) in tsets.items():
            rnd = np.array([metrics(ps, yv, rr.sample(allh, K)) for _ in range(30)])
            ablation["sets"][name] = dict(n=len(ps), base=metrics(ps, yv), top1=metrics(ps, yv, topK[:1]), top2=metrics(ps, yv, topK[:2]),
                topK=metrics(ps, yv, topK), todas=metrics(ps, yv, allh),
                random=[round(float(rnd[:, 0].mean()), 3), round(float(rnd[:, 0].std()), 3), round(float(rnd[:, 1].mean()), 2), round(float(rnd[:, 1].std()), 2)])
        jr = {}
        for cw, kw, cfem in [(F, M, True), (M, F, False)]:
            ic, ik = enc(cw), enc(kw); lc, _, hc2 = forward(ic, rec=True); dc, dk = ld(lc), ld(forward(ik)[0])
            good = ((dc > 0) == cfem) & ((dk > 0) != cfem) & ((dc - dk).abs() > MIN_GAP)
            if int(good.sum()) == 0: continue
            for kk in sorted({1, min(2, K), K}):
                lp = forward(ik, hi=mk(topK[:kk]), src=hc2)[0]; jr.setdefault(kk, []).extend(((ld(lp) - dk) / (dc - dk))[good].cpu().numpy().tolist())
        ablation["joint_recovery"] = {str(k_): round(float(np.mean(v)), 3) for k_, v in jr.items()}
    print("  ablación lista" if ablation else "  sin ablación")
    return dict(model=MODEL, layers=nl, nheads=H, pos=["sustantivo", "es"], adj=adj, n_words=len(ws), n_pairs=N, beh=beh, margins=margins,
                nouns=nouns, cue=cue, patch=patch, heads=heads, probe=probe, ablation=ablation)

results = []
for m in MODELS:
    try: results.append(run_model(m))
    except Exception as e:
        import traceback; traceback.print_exc(); results.append(dict(model=m, error=repr(e)))
    gc.collect()
    if dev == "cuda": torch.cuda.empty_cache()
js = json.dumps(dict(models=results), ensure_ascii=False, separators=(",", ":")); open("atlas_v4.json", "w", encoding="utf-8").write(js)
for r in results:
    if "error" not in r and r["heads"]: print(r["model"], "cabezas más influyentes (capa, cabeza, recuperación):", r["heads"]["top"][:6])
print("\nListo. Se ha guardado atlas_v4.json (en Colab también se descarga solo). Si no puedes subirlo, copia esta línea entera:\n" + js)
try:
    from google.colab import files; files.download("atlas_v4.json")
except Exception: pass
for r in results:
    if "error" in r or not r.get("ablation"): continue
    a = r["ablation"]; print(r["model"], "K =", a["K"], "cabezas", a["heads"], "recuperación conjunta", a.get("joint_recovery"))
    for k, v in a["sets"].items(): print("  ", k, "| base", v["base"][0], "| sin cabezas clave", v["topK"][0], "| al azar", v["random"][0], "| sin ninguna", v["todas"][0])
