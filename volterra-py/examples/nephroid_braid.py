import volterra as v

DIVISOR = 0.764031 * 100.0

mesh = v.confined_mesh(
    v.PlaneCurve.epitrochoid(q=2.0, d=1.0, r=49.778694002),
    h_bulk=2.0, h_min=2.0, cusp_edge=2.0,
)

run = v.ConfinedRun(
    mesh,
    active_length=0.0128 * DIVISOR,
    coherence_length=0.0766 * DIVISOR,
    resolution=100,
    q_anchor=1.0,
    wall="noslip",
    dt=1e-4,
    seed=0,
)

frames = []
for _ in range(100):
    run.step(200)
    frames.append(run.defects())

settled = next(
    i for i in range(len(frames))
    if all(sum(c > 0 for _, _, c in f) == 4 for f in frames[i:])
)
word = v.BraidWord.from_frames(frames[settled:])
block = word.fundamental_period()[-6:]

print(block)
print(v.BraidWord(4, block).entropy())
