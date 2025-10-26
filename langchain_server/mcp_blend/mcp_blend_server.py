from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import pulp

class MCPHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        data = self.rfile.read(content_length)
        req = json.loads(data)
        tool = req.get("tool_id")
        inp = req.get("input", {})

        if tool == "optimize_blend":
            oils = inp.get("oils", [])
            demand = inp.get("demand", 1000)
            cost = {o["name"]: o["cost"] for o in oils}
            iodine = {o["name"]: o["iodine"] for o in oils}

            # LP問題設定：コスト最小化
            prob = pulp.LpProblem("BlendOptimization", pulp.LpMinimize)
            vars = {name: pulp.LpVariable(name, lowBound=0) for name in cost.keys()}
            prob += pulp.lpSum([cost[i]*vars[i] for i in vars])
            prob += pulp.lpSum([vars[i] for i in vars]) == demand
            prob += pulp.lpSum([iodine[i]*vars[i] for i in vars]) / demand >= 100
            prob.solve(pulp.PULP_CBC_CMD(msg=False))

            result = {i: vars[i].value() for i in vars}
            total_cost = pulp.value(prob.objective)

            resp = {"output": {"blend": result, "total_cost": total_cost}}
        else:
            resp = {"error": f"Unknown tool {tool}"}

        body = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

def run():
    server = HTTPServer(("0.0.0.0", 9100), MCPHandler)
    print("MCP blend optimizer running on port 9100")
    server.serve_forever()

if __name__ == "__main__":
    run()
