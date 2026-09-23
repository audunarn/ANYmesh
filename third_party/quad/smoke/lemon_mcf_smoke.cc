// Minimal, self-contained LEMON min-cost-flow (NetworkSimplex) smoke test.
//
// Exercises the exact LEMON routine that libSatsuma's
// solve_mcf_via_lemon_netsimp() drives, against the pinned + vendored LEMON
// sources, to prove they compile and compute a correct min-cost flow with
// bare cl.exe + Windows SDK.
//
// Graph (nodes 0..3), unit capacities:
//    0 --1--> 1 --1--> 3
//    0 --3--> 2 --1--> 3
//  supply: 0 = +2 (source), 3 = -2 (sink); others 0.
//  Cheapest 2 units: path 0->1->3 (cost 2) and 0->2->3 (cost 4).
//  total cost = 6.
//
// Expected: status OPTIMAL, total cost 6.

#include <cstdio>
#include <climits>
#include <lemon/list_graph.h>
#include <lemon/network_simplex.h>

using namespace lemon;

int main()
{
    typedef ListDigraph G;
    typedef G::Node Node;
    typedef G::Arc Arc;

    G g;
    Node s = g.addNode();
    Node a = g.addNode();
    Node b = g.addNode();
    Node t = g.addNode();

    Arc sa = g.addArc(s, a);
    Arc sb = g.addArc(s, b);
    Arc at = g.addArc(a, t);
    Arc bt = g.addArc(b, t);

    const int INF = INT_MAX;
    G::NodeMap<int> supply(g);
    supply[s] = 2;
    supply[a] = 0;
    supply[b] = 0;
    supply[t] = -2;

    G::ArcMap<int> cap(g);
    cap[sa] = cap[sb] = cap[at] = cap[bt] = 1;   // unit capacities

    G::ArcMap<int> cost(g);
    cost[sa] = 1; cost[sb] = 3; cost[at] = 1; cost[bt] = 1;

    NetworkSimplex<G> ns(g);
    ns.supplyMap(supply).upperMap(cap).costMap(cost);

    NetworkSimplex<G>::ProblemType res = ns.run();

    int total = ns.totalCost<int>();

    const bool ok = (res == NetworkSimplex<G>::OPTIMAL) && (total == 6);
    std::printf("lemon_mcf_smoke: status=%d flow=(%d,%d,%d,%d) total_cost=%d expected=6 -> %s\n",
                static_cast<int>(res),
                ns.flow(sa), ns.flow(sb), ns.flow(at), ns.flow(bt),
                total, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
