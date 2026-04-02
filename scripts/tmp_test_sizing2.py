from agents.risk_agent import RiskManagerAgent
import config

# Manually override the config just for this test to prove the 2.0x cap works
config.RISK_CONFIG["max_leverage"] = 2.0

r = RiskManagerAgent()

res = r.evaluate(
    [{'ticker':'XYZ','direction':'long','confidence':0.85,'current_price':65.55,'stop_loss':0,'atr_5m':0.10}],
    {'equity':97000,'buying_power':387000},
    [],
    20
)
t = res['approved_trades']
print('=== MAX LEV SCENARIO (LIVE) ===')
if t:
    lev = t[0].get('leverage', '?')
    print(f"qty={t[0]['qty']} shares, deploy=${t[0]['position_dollars']:,.0f}, leverage={lev}x, pct_of_equity={t[0]['position_pct']:.1f}%")
else:
    print('REJECTED:', res['rejected_trades'])
