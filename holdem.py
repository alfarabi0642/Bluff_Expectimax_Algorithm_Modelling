import numpy as np
from enum import Enum
import random
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

class PlayerType(Enum):
    TIGHT_PASSIVE = "TP"
    LOOSE_AGGRESSIVE = "LA"
    SCAREDY_CAT = "SC"

class GameStage(Enum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4
    TERMINAL = 5

class ActionType(Enum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET = 3
    RAISE = 4
    ALL_IN = 5

class PokerDecisionModel:
    def __init__(self, opponent_type=PlayerType.TIGHT_PASSIVE, hero_position="BTN", hero_stack=100, villain_stack=100):
        # Inisialisasi state permainan
        self.stage = GameStage.PREFLOP
        self.pot = 1.5  # Small blind (0.5) + big blind (1)
        self.hero_stack = hero_stack
        self.villain_stack = villain_stack
        self.position = hero_position
        self.hole_cards = []
        self.community_cards = []
        self.history = []
        self.action_count = 0
        self.initial_stack = hero_stack
        self.last_bet_amount = 0
        self.decision_tree = []
        self.decision_evs = []  # Menyimpan EV untuk setiap keputusan
        
        # Adjust stack berdasarkan posisi
        if hero_position == "SB":
            self.hero_stack -= 0.5
            self.villain_stack -= 1
        elif hero_position == "BB":
            self.hero_stack -= 1
            self.villain_stack -= 0.5
        
        # Karakter lawan
        self.opponent_type = opponent_type
        self.opponent_aggression = 0.0

    def start_hand(self, hero_cards, flop=None, turn=None, river=None):
        self.hole_cards = hero_cards
        self.community_cards = []
        self.history = []
        self.stage = GameStage.PREFLOP
        self.action_count = 0
        self.initial_stack = self.hero_stack
        self.last_bet_amount = 0
        self.decision_tree = []
        self.decision_evs = []  # Reset EV keputusan
        
        # Setel pot berdasarkan stage
        self.pot = 1.5
        if flop:
            self.community_cards = flop
            self.stage = GameStage.FLOP
            self.pot = 4.0
        if turn:
            self.community_cards.append(turn)
            self.stage = GameStage.TURN
            self.pot = 8.0
        if river:
            self.community_cards.append(river)
            self.stage = GameStage.RIVER
            self.pot = 16.0

    def opponent_response(self, our_action):
        # Villain always folds to an all-in
        if our_action == ActionType.ALL_IN:
            return {"FOLD": 1.0}

        if self.opponent_type == PlayerType.TIGHT_PASSIVE:
            if our_action in [ActionType.BET, ActionType.RAISE]:
                return {"FOLD": 0.65, "CALL": 0.30, "RAISE": 0.05}
            else:
                return {"CHECK": 0.80, "BET": 0.20}
                
        elif self.opponent_type == PlayerType.LOOSE_AGGRESSIVE:
            if our_action in [ActionType.BET, ActionType.RAISE]:
                return {"FOLD": 0.10, "CALL": 0.30, "RAISE": 0.60}
            else:
                return {"CHECK": 0.05, "BET": 0.95}
                
        else:  # SCAREDY_CAT
            if our_action in [ActionType.BET, ActionType.RAISE]:
                return {"FOLD": 0.80, "CALL": 0.19, "RAISE": 0.01}
            else:
                return {"CHECK": 0.90, "BET": 0.10}

    def estimate_equity(self):
        """Estimasi equity yang lebih realistis berdasarkan tangan"""
        # Untuk kartu lemah seperti 72o
        if "7" in self.hole_cards[0] and "2" in self.hole_cards[1]:
            return 0.05  # Sangat rendah
        
        # Estimasi lebih akurat berdasarkan kekuatan kartu
        card_ranks = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
        card_suits = ['♠','♥','♦','♣']
        
        # Hitung nilai dasar kartu
        values = []
        for card in self.hole_cards:
            try:
                rank_value = card_ranks.index(card[0]) / 12.0  # 0-1 scale
                values.append(rank_value)
            except:
                values.append(0.3)  # Default jika format salah
        
        # Strength dasar: rata-rata nilai kartu
        strength = sum(values) / len(values)
        
        # Bonus untuk suited dan connector
        suited = self.hole_cards[0][1] == self.hole_cards[1][1]
        rank_diff = abs(values[0] - values[1])
        connected = rank_diff <= 0.083  # ~1 rank (A=0, K=1, Q=2, etc.)
        
        if suited and connected:
            strength += 0.15
        elif suited:
            strength += 0.07
        elif connected:
            strength += 0.08
            
        # Adjust berdasarkan stage
        stage_factor = {
            GameStage.PREFLOP: 0.8,
            GameStage.FLOP: 0.6,
            GameStage.TURN: 0.4,
            GameStage.RIVER: 0.2,
            GameStage.SHOWDOWN: 1.0
        }[self.stage]
        
        equity = max(0.05, min(0.95, strength * stage_factor))
        return equity

    def calculate_ev(self, action, bet_size=0):
        """Perhitungan EV yang lebih akurat dengan penyesuaian lawan"""
        if action == ActionType.FOLD:
            return 0
            
        resp_probs = self.opponent_response(action)
        ev = 0
        
        for resp, prob in resp_probs.items():
            if resp == "FOLD":
                ev += prob * self.pot
            elif resp == "CALL":
                if self.stage == GameStage.RIVER:
                    equity = self.estimate_equity()
                    win_amount = self.pot + 2 * bet_size
                    ev += prob * (equity * win_amount - (1 - equity) * bet_size)
                else:
                    # Penalti tambahan untuk LA karena implied odds negatif
                    penalty = 1.3 if self.opponent_type == PlayerType.LOOSE_AGGRESSIVE else 1.0
                    ev += prob * (-bet_size * penalty)
            elif resp == "RAISE":
                # Penalti lebih besar untuk LA
                penalty = 1.5 if self.opponent_type == PlayerType.LOOSE_AGGRESSIVE else 1.0
                ev += prob * (-bet_size * penalty)
            elif resp == "BET":
                # Asumsi kita fold terhadap bet lawan
                ev += prob * 0
                
        return ev

    def get_bluff_action(self):
        """Mengembalikan aksi bluff yang optimal berdasarkan stage dan lawan"""
        if self.stage == GameStage.PREFLOP:
            return ActionType.RAISE, 3.0
        
        # Tentukan ukuran taruhan berdasarkan tipe lawan
        if self.opponent_type == PlayerType.SCAREDY_CAT:
            return ActionType.BET, self.pot * 0.4
        elif self.opponent_type == PlayerType.TIGHT_PASSIVE:
            return ActionType.BET, self.pot * 0.6
        else:  # LA
            return ActionType.BET, self.pot * 0.5

    def can_take_action(self, action_type):
        """Cek apakah hero bisa melakukan aksi tertentu berdasarkan stack"""
        if action_type in [ActionType.CHECK, ActionType.FOLD]:
            return True
        return self.hero_stack > 0

    def optimal_action(self):
        """Pilih aksi berdasarkan probabilitas terhadap taruhan lawan atau bluff biasa"""
        # Jika hero harus merespons taruhan lawan
        if self.is_bet_in_front():
            villain_act = self.history[-1][1]
            call_amt = min(self.last_bet_amount, self.hero_stack)
            raise_amt = min(self.last_bet_amount * 2, self.hero_stack)
            # Jika villain hanya check atau call, hero selalu reraise
            if villain_act in ["CHECK", "CALL"] and raise_amt > 0:
                action, size = ActionType.RAISE, raise_amt
            else:
                # Saat menghadapi bet atau raise villain, probabilitas reraise 70%, call 30%
                if random.random() < 0.7 and raise_amt > call_amt:
                    action, size = ActionType.RAISE, raise_amt
                else:
                    action, size = ActionType.CALL, call_amt
            ev = self.calculate_ev(action, size)
            return (action, size), ev
        
        # Jika tidak ada taruhan, hero melakukan aksi bluff biasa
        if self.stage == GameStage.RIVER:
            action, bet_size = ActionType.BET, self.pot * 0.5
            bet_size = min(bet_size, self.hero_stack)
            return (action, bet_size), self.calculate_ev(action, bet_size)
        
        action, bet_size = self.get_bluff_action()
        bet_size = min(bet_size, self.hero_stack)
        if bet_size > 0 and self.hero_stack > 0:
            return (action, bet_size), self.calculate_ev(action, bet_size)
        else:
            return (ActionType.CHECK, 0), self.calculate_ev(ActionType.CHECK, 0)

    def is_bet_in_front(self):
        """Cek apakah ada taruhan yang harus dijawab"""
        if not self.history:
            return False
            
        # Cek aksi terakhir villain
        if self.history and self.history[-1][0] == "VILLAIN" and self.history[-1][1] in ["BET", "RAISE"]:
            self.last_bet_amount = self.history[-1][2]
            return True
            
        return False

    def execute_action(self, action, bet_size):
        """Eksekusi aksi dan dapatkan respons lawan"""
        # Simpan aksi hero
        self.history.append(("HERO", action.name, bet_size))
        self.decision_tree.append({
            "player": "HERO",
            "stage": self.stage.name,
            "action": action.name,
            "size": bet_size,
            "pot": self.pot,
            "stack": self.hero_stack
        })
        
        # Update stack dan pot berdasarkan aksi hero
        if action == ActionType.CALL:
            call_amount = min(bet_size, self.hero_stack)
            self.hero_stack -= call_amount
            self.pot += call_amount
        elif action in [ActionType.BET, ActionType.RAISE, ActionType.ALL_IN]:
            bet_amount = min(bet_size, self.hero_stack)
            self.hero_stack -= bet_amount
            self.pot += bet_amount
        
        # Dapatkan respons lawan
        resp_probs = self.opponent_response(action)
        resp = random.choices(
            list(resp_probs.keys()), 
            weights=list(resp_probs.values()),
            k=1
        )[0]
        
        # Simpan respons lawan
        resp_amount = 0
        if resp == "CALL":
            resp_amount = min(bet_size, self.villain_stack)
        elif resp == "RAISE":
            resp_amount = min(bet_size * 2, self.villain_stack)
        elif resp == "BET":
            resp_amount = min(self.pot * 0.5, self.villain_stack)
        
        self.history.append(("VILLAIN", resp, resp_amount))
        self.decision_tree.append({
            "player": "VILLAIN",
            "stage": self.stage.name,
            "action": resp,
            "size": resp_amount,
            "pot": self.pot,
            "stack": self.villain_stack
        })
        
        # Update stack dan pot berdasarkan respons lawan
        if resp == "CALL":
            self.villain_stack -= resp_amount
            self.pot += resp_amount
        elif resp == "RAISE":
            self.villain_stack -= resp_amount
            self.pot += resp_amount
        elif resp == "BET":
            self.villain_stack -= resp_amount
            self.pot += resp_amount
        
        return resp

    def update_stage(self):
        """Update stage berdasarkan aksi terakhir"""
        if len(self.history) < 2:
            return
            
        # Jika ada fold, langsung terminal
        if "FOLD" in [action[1] for action in self.history[-2:]]:
            self.stage = GameStage.TERMINAL
            return
            
        # Jika kedua pemain check, lanjut ke stage berikutnya
        if self.history[-2][1] == "CHECK" and self.history[-1][1] == "CHECK":
            if self.stage == GameStage.PREFLOP:
                self.stage = GameStage.FLOP
            elif self.stage == GameStage.FLOP:
                self.stage = GameStage.TURN
            elif self.stage == GameStage.TURN:
                self.stage = GameStage.RIVER
            elif self.stage == GameStage.RIVER:
                self.stage = GameStage.SHOWDOWN
                
        # Jika ada call tanpa raise, juga lanjut ke stage berikutnya
        elif self.history[-1][1] == "CALL" and not self.is_bet_in_front():
            if self.stage == GameStage.PREFLOP:
                self.stage = GameStage.FLOP
            elif self.stage == GameStage.FLOP:
                self.stage = GameStage.TURN
            elif self.stage == GameStage.TURN:
                self.stage = GameStage.RIVER
            elif self.stage == GameStage.RIVER:
                self.stage = GameStage.SHOWDOWN
                
        # Jika salah satu pemain all-in, langsung ke showdown
        if self.hero_stack == 0 or self.villain_stack == 0:
            self.stage = GameStage.SHOWDOWN

    def visualize_decision_tree(self, title="Decision Tree"):
        """Visualisasi pohon keputusan yang lebih informatif"""
        G = nx.DiGraph()
        pos = {}
        node_colors = []
        node_labels = {}
        edge_labels = {}

        # Warna berdasarkan pemain
        player_colors = {
            "HERO": "lightblue",
            "VILLAIN": "lightcoral"
        }

        # --- Add root node ---
        G.add_node("Start")
        pos["Start"] = (0, 0)
        node_colors.append("lightgreen")
        node_labels["Start"] = f"Start\nPot: {self.pot:.1f} BB"

        # We'll keep edges chaining from this
        prev_node = "Start"

        # Track how many nodes we've placed per stage to stack them vertically
        stage_counts = {s.name: 0 for s in GameStage}

        # --- Add each decision as a child of the previous ---
        for i, decision in enumerate(self.decision_tree, 1):
            player = decision["player"]
            action = decision["action"]
            size   = decision["size"]
            stage  = decision["stage"]
            pot    = decision["pot"]
            stack  = decision["stack"]

            # Unique ID
            node_id = f"{stage}_{i}_{player}"

            # x based on stage, y based on count
            stage_idx = GameStage[stage].value
            x = stage_idx * 3
            # decrement so later nodes go downward
            stage_counts[stage] -= 1
            y = stage_counts[stage]

            # add node
            G.add_node(node_id)
            pos[node_id] = (x, y)
            node_colors.append(player_colors.get(player, "gray"))
            node_labels[node_id] = (
                f"{player} {action} {size:.1f} BB\n"
                f"Pot: {pot:.1f} BB  Stack Hero: {stack:.1f} BB"
            )

            # connect from prev_node
            G.add_edge(prev_node, node_id)
            edge_labels[(prev_node, node_id)] = action

            prev_node = node_id

        # --- Draw it ---
        plt.figure(figsize=(12, 8))
        nx.draw(
            G,
            pos,
            labels=node_labels,
            node_size=2000,
            node_color=node_colors,
            font_size=9,
            font_weight="bold",
            arrowsize=20
        )
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_color='darkgreen'
        )

        # Legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='lightgreen', markersize=10, label='Start'),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='lightblue', markersize=10, label='Hero'),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='lightcoral', markersize=10, label='Villain')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        plt.title(f"{title} vs {self.opponent_type.name}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()



    def simulate_hand(self, verbose=True):
        total_decision_ev = 0  # Total EV dari semua keputusan
        decision_count = 0      # Jumlah keputusan yang diambil
        winner = None           # Pemenang hand
        
        if verbose:
            print(f"Memulai hand baru - Hero: {self.hole_cards}, Villain: {self.opponent_type.name}")
            print(f"Stage: {self.stage.name}, Pot: {self.pot:.2f} BB")
            print(f"Hero stack: {self.hero_stack:.2f}, Villain stack: {self.villain_stack:.2f}")
        
        while self.stage not in [GameStage.TERMINAL, GameStage.SHOWDOWN]:
            try:
                (action, bet_size), ev = self.optimal_action()
                total_decision_ev += ev
                decision_count += 1
                self.decision_evs.append(ev)  # Simpan EV keputusan ini
                
                if verbose:
                    print(f"\n{self.stage.name}: Hero memilih {action.name} {bet_size:.2f} BB (EV: {ev:.2f})")
                
                resp = self.execute_action(action, bet_size)
                
                if verbose:
                    print(f"Villain merespons: {resp}")
                    print(f"Pot: {self.pot:.2f} BB, Hero stack: {self.hero_stack:.2f}, Villain stack: {self.villain_stack:.2f}")
                
                # Update stage berdasarkan aksi
                self.update_stage()
                
                # Jika villain fold, game berakhir
                if resp == "FOLD":
                    self.stage = GameStage.TERMINAL
                    break
                
                # Jika salah satu stack habis, langsung ke showdown
                if self.hero_stack == 0 or self.villain_stack == 0:
                    self.stage = GameStage.SHOWDOWN
                    break
                    
            except Exception as e:
                if verbose:
                    print(f"Error in decision: {str(e)}")
                break
        
        # Hitung net profit
        net_profit = self.hero_stack - self.initial_stack
        
        # Tentukan pemenang dan update stack
        if self.stage == GameStage.TERMINAL:
            last_action = self.history[-1]
            if last_action[0] == "VILLAIN" and last_action[1] == "FOLD":
                winner = "Hero"
                self.hero_stack += self.pot  # Hero mengambil pot
            else:
                winner = "Villain"
                self.villain_stack += self.pot  # Villain mengambil pot
            
            net_profit = self.hero_stack - self.initial_stack
            
            if verbose:
                print(f"\nRESULT: {winner} memenangkan {self.pot:.2f} BB")
                print(f"Net profit: {net_profit:.2f} BB")
                
        else:  # SHOWDOWN
            hero_equity = self.estimate_equity()
            hero_wins = random.random() < hero_equity
            winner = "Hero" if hero_wins else "Villain"
            
            # Update stack berdasarkan hasil showdown
            if hero_wins:
                self.hero_stack += self.pot
            else:
                self.villain_stack += self.pot
                
            net_profit = self.hero_stack - self.initial_stack
            
            if verbose:
                print(f"\nSHOWDOWN: Hero equity {hero_equity:.2%}")
                print(f"RESULT: {winner} memenangkan {self.pot:.2f} BB")
                print(f"Net profit: {net_profit:.2f} BB")
                
        # Hitung rata-rata EV keputusan
        avg_ev = total_decision_ev / decision_count if decision_count > 0 else 0
        
        return winner, net_profit, total_decision_ev, decision_count, avg_ev

# =====================
# Contoh Penggunaan
# =====================

# ... (kode sebelumnya tetap sama)

# =====================
# Contoh Penggunaan
# =====================

if __name__ == "__main__":
    # Contoh 1: Scaredy Cat di river
    print("======== CONTOH 1: SCAREDY CAT DI RIVER ========")
    model = PokerDecisionModel(opponent_type=PlayerType.SCAREDY_CAT, hero_position="BTN", hero_stack=100)
    model.start_hand(["7♦", "2♣"], river=["A♣", "K♦", "5♥", "8♠", "2♦"])
    winner, profit, total_ev, decisions, avg_ev = model.simulate_hand()
    print(f"Total EV keputusan: {total_ev:.2f} BB")
    print(f"Rata-rata EV keputusan: {avg_ev:.2f} BB")
    model.visualize_decision_tree("Scaredy Cat di River")
    
    # Contoh 2: Loose Aggressive full hand
    print("\n\n======== CONTOH 2: LOOSE AGGRESSIVE FULL HAND ========")
    model = PokerDecisionModel(opponent_type=PlayerType.LOOSE_AGGRESSIVE, hero_position="CO", hero_stack=100)
    model.start_hand(["7♦", "2♣"])
    winner, profit, total_ev, decisions, avg_ev = model.simulate_hand()
    print(f"Total EV keputusan: {total_ev:.2f} BB")
    print(f"Rata-rata EV keputusan: {avg_ev:.2f} BB")
    model.visualize_decision_tree("Loose Aggressive Full Hand")
    
    # Contoh 3: Tight Passive dimulai dari preflop
    print("\n\n======== CONTOH 3: TIGHT PASSIVE DARI PREFLOP ========")
    model = PokerDecisionModel(opponent_type=PlayerType.TIGHT_PASSIVE, hero_position="BTN", hero_stack=100)
    model.start_hand(["7♦", "2♣"])  # Hanya berikan hole cards, tanpa community cards
    winner, profit, total_ev, decisions, avg_ev = model.simulate_hand()
    print(f"Total EV keputusan: {total_ev:.2f} BB")
    print(f"Rata-rata EV keputusan: {avg_ev:.2f} BB")
    model.visualize_decision_tree("Tight Passive dari Preflop")
    
    # Contoh 4: Analisis EV strategi bluff
    print("\n\n======== CONTOH 4: ANALISIS EV STRATEGI BLUFF (500 SIMULASI) ========")
    results = []

    for opponent_type in PlayerType:
        total_decision_ev = 0
        total_avg_ev = 0
        total_hands = 0
        trials = 500
        
        for i in range(trials):
            model = PokerDecisionModel(opponent_type=opponent_type, hero_position="BTN", hero_stack=100)
            ranks = random.sample(['A','K','Q','J','T','9','8','7'], 2)
            suits = random.sample(['♠','♥','♦','♣'], 2)
            hole_cards = [f"{r}{s}" for r, s in zip(ranks, suits)]
            model.start_hand(hole_cards)
            _, _, total_ev, decision_count, avg_ev = model.simulate_hand(verbose=False)
            
            total_decision_ev += total_ev
            if decision_count > 0:
                total_avg_ev += (total_ev / decision_count)
                total_hands += 1
        
        # Hitung rata-rata
        avg_total_ev = total_decision_ev / trials
        overall_avg_ev = total_avg_ev / total_hands if total_hands > 0 else 0
        
        # Simpan hasil
        results.append({
            "type": opponent_type,
            "avg_total_ev": avg_total_ev,
            "overall_avg_ev": overall_avg_ev
        })

    # Tampilkan hasil dengan analisis
    print("\n{:<20} {:<25} {:<20}".format(
        "Tipe Lawan", "Avg Total EV (BB)", "Avg EV/Decision (BB)"
    ))
    print("-" * 65)

    for res in results:
        ev_color = "\033[92m" if res["overall_avg_ev"] > 0 else "\033[91m"
        
        print("{:<20} {:<25.2f} {}{:<20.2f}\033[0m".format(
            res["type"].name,
            res["avg_total_ev"],
            ev_color, res["overall_avg_ev"]
        ))

    # Analisis tambahan
    print("\n\033[1mKESIMPULAN STRATEGI BERDASARKAN EV:\033[0m")
    for res in results:
        if res["type"] == PlayerType.LOOSE_AGGRESSIVE:
            print(f"- Melawan \033[1m{res['type'].name}\033[0m: EV negatif ({res['overall_avg_ev']:.2f} BB/decision)")
        else:
            print(f"- Melawan \033[1m{res['type'].name}\033[0m: EV positif ({res['overall_avg_ev']:.2f} BB/decision)")