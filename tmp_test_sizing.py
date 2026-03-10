from agents.risk_agent import RiskManagerAgent
r = RiskManagerAgent()

# Simulate XYZ scenario that blew up: $97k equity, $387k BP, ATR=$0.87, price=$65.55
res = r.evaluate(
    [{'ticker':'XYZ','direction':'long','confidence':0.85,'current_price':65.55,'stop_loss':64.25,'atr_5m':0.87}],
    {'equity':97000,'buying_power':387000},
    [],
    20
)

t = res['approved_trades']
print('=== XYZ BLOW-UP SCENARIO (stop=$64.25, ATR=$0.87) ===')
if t:
    lev = t[0].get('leverage', '?')
    print(f"qty={t[0]['qty']} shares, deploy=${t[0]['position_dollars']:,.0f}, leverage={lev}x, pct_of_equity={t[0]['position_pct']:.1f}%")
else:
    print('REJECTED:', res['rejected_trades'])

# Simulate tiny ATR (worst case: max leverage hit)
res2 = r.evaluate(
    [{'ticker':'XYZ','direction':'long','confidence':0.85,'current_price':65.55,'stop_loss':0,'atr_5m':0.10}],
    {'equity':97000,'buying_power':387000},
    [],
    20
)
t2 = res2['approved_trades']
print('\n=== TINY ATR=$0.10 (max leverage scenario) ===')
if t2:
    lev2 = t2[0].get('leverage', '?')
    print(f"qty={t2[0]['qty']} shares, deploy=${t2[0]['position_dollars']:,.0f}, leverage={lev2}x, pct_of_equity={t2[0]['position_pct']:.1f}%")
else:
    print('REJECTED:', res2['rejected_trades'])

# Simulate normal case: wider ATR
res3 = r.evaluate(
    [{'ticker':'XYZ','direction':'long','confidence':0.85,'current_price':65.55,'stop_loss':64.25,'atr_5m':3.0}],
    {'equity':97000,'buying_power':387000},
    [],
    20
)
t3 = res3['approved_trades']
print('\n=== NORMAL ATR=$3.0 (comfortable stop) ===')
if t3:
    lev3 = t3[0].get('leverage', '?')
    print(f"qty={t3[0]['qty']} shares, deploy=${t3[0]['position_dollars']:,.0f}, leverage={lev3}x, pct_of_equity={t3[0]['position_pct']:.1f}%")
else:
    print('REJECTED:', res3['rejected_trades'])
