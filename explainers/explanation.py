class Explanation:
    """
    Base class for explanations.
    """

    def __init__(self, explanation_type: str, sentence: str, tokens: list):
        self.explanation_type: str = explanation_type
        self.sentence: str = sentence
        self.tokens: list = tokens
        self.scores: dict = {}

    def add_token(self, scores: dict, token: str, position: int):
        if position not in self.scores:
            self.scores[position] = {
                "token": token,
                "intp": {},
            }

        for sub_position, sub_score in scores.items():
            if sub_position == position:
                continue # skip self-influence
            
            if sub_position not in self.scores[position]["intp"]:
                self.scores[position]["intp"][sub_position] = {
                    "token": sub_score["token"],
                    "score": sub_score["score"],
                }
            else:
                # If the sub_position already exists, we can update the score
                self.scores[position]["intp"][sub_position]["score"] += sub_score[
                    "score"
                ]
    def normalize(self):
        """
        Normalize scores for each token's influences to the range [0, 1].
        """
        for main_pos, main_data in self.scores.items():
            influences = main_data["intp"]
            if not influences:
                continue
            
            scores = [abs(data["score"]) for data in influences.values()]
            max_score = max(scores)
            min_score = min(scores)

            # Normalize scores to [0, 1]
            padd = 0
            if min_score < 0:
                padd = abs(min_score)
            for sub_pos, data in influences.items():
                if max_score == 0:
                    normalized_score = 0
                else:
                    normalized_score =  (padd + abs(data["score"])) /  (padd + max_score)
                influences[sub_pos]["score"] = normalized_score
                
        return self
    def add_one_word(
        self, main_token, main_position: int, sub_token, sub_position: int, score: float
    ):
        if  main_position == sub_position:
            return  # Skip self-influence
        
        if main_position not in self.scores:
            self.scores[main_position] = {
                "token": main_token,
                "intp": {sub_position: {"token": sub_token, "score": score}},
            }
        else:
            self.scores[main_position]["intp"][sub_position] = {
                "token": sub_token,
                "score": score,
            }

    def __str__(self):
        return f"Explanation(type={self.explanation_type})"

    def __repr__(self):
        return self.__str__()

    def plot_one(self, token_pos, save_path=None, show=False):
        """
        Plot a bar chart showing how each token influences the token at position token_pos.

        Args:
            token_pos: Position of the token to explain
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if token_pos not in self.scores:
            raise ValueError(f"No explanation data for token at position {token_pos}")

        token_data = self.scores[token_pos]
        token = token_data["token"]
        influences = token_data["intp"]

        positions = []
        tokens = []
        scores = []

        for pos, data in influences.items():
            if pos == token_pos:
                continue
            positions.append(pos)
            tokens.append(data["token"])
            scores.append(data["score"])

        sorted_indices = np.argsort(positions)
        positions = [positions[i] for i in sorted_indices]
        tokens = [tokens[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        plt.figure(figsize=(12, 6))

        scores = np.abs(scores)

        if len(scores) > 0 and max(scores) > 0:
            scores = scores / max(scores)

        bars = plt.bar(range(len(tokens)), scores)

        insert_index = 0
        while insert_index < len(positions) and positions[insert_index] < token_pos:
            insert_index += 1

        plt.axvline(
            x=insert_index - 0.5,
            color="blue",
            linestyle="--",
            alpha=0.7,
            label=f'Position of "{token}" (analyzed)',
        )
        plt.legend()

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height if height >= 0 else height - 0.3,
                f"{score:.2f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=9,
            )

        # Set x-axis labels to tokens
        plt.xticks(range(len(tokens)), tokens, rotation=90)

        # Add title and labels
        plt.title(
            f"Influence on token \"{token}\" at position {token_pos} in the sentence:\n'{self.sentence}' with {self.explanation_type}",
            fontsize=14,
        )
        plt.ylabel("Influence Score", fontsize=12)
        plt.xlabel("Tokens", fontsize=12)

        # Add a horizontal line at y=0
        plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
        # Adjust layout and display
        plt.tight_layout()
        if show:
            plt.show()

        return plt

    def plot_comparison(self, token_pos, *args, save_path=None, show=False, title=None):
        """
        Plot a comparison of multiple explanation methods for the same token.

        Args:
            token_pos: Position of the token to explain
            *args: Additional Explanation objects to compare with
            save_path: Optional path to save the figure
            show: Whether to display the plot
            title: Custom title for the plot
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Check if the token position exists in scores
        if token_pos not in self.scores:
            raise ValueError(f"No explanation data for token at position {token_pos}")

        # Get the token and its influence scores from this explanation
        token_data = self.scores[token_pos]
        token = token_data["token"]

        # Create a list with all explanations (self + args)
        all_explanations = [self] + list(args)
        explanation_types = [exp.explanation_type for exp in all_explanations]

        # Check if all explanations have data for this token
        for i, exp in enumerate(all_explanations):
            if token_pos not in exp.scores:
                raise ValueError(
                    f"Explanation {explanation_types[i]} has no data for token at position {token_pos}"
                )

        # Get all tokens that appear in any explanation
        all_tokens = set()
        all_positions = set()

        for exp in all_explanations:
            influences = exp.scores[token_pos]["intp"]
            for pos, data in influences.items():
                if pos != token_pos:  # Skip self-influence
                    all_tokens.add(data["token"])
                    all_positions.add(pos)

        # Convert to sorted lists
        positions = sorted(list(all_positions))
        tokens = [all_explanations[0].tokens[pos] for pos in positions]

        # Create a figure
        fig, ax = plt.subplots(figsize=(max(12, len(tokens) * 0.8), 8))

        # Number of explanation methods
        n_explanations = len(all_explanations)

        # Width of each bar group
        group_width = 0.8
        # Width of each bar within a group
        bar_width = group_width / n_explanations

        # Colors for different explanation methods

        # For each explanation method, plot a set of bars
        for i, exp in enumerate(all_explanations):
            # Get scores for this explanation
            scores = []
            for pos in positions:
                if pos in exp.scores[token_pos]["intp"]:
                    scores.append(exp.scores[token_pos]["intp"][pos]["score"])
                else:
                    scores.append(0)  # No score available

            # Convert to absolute values and normalize
            scores = np.abs(np.array(scores))
            if np.max(scores) > 0:
                scores = scores / np.max(scores)

            # Calculate bar positions
            x = np.arange(len(tokens))
            bar_positions = x - group_width / 2 + (i + 0.5) * bar_width

            # Plot bars
            bars = ax.bar(
                bar_positions, scores, width=bar_width, label=exp.explanation_type
            )

            # Add value labels on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                if height > 0.05:  # Only label bars with significant height
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.02,
                        f"{score:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90 if n_explanations > 2 else 0,
                    )

        # Find where to place the vertical line based on the token's position
        if token_pos in positions:
            token_idx = positions.index(token_pos)
            ax.axvline(
                x=token_idx,
                color="blue",
                linestyle="--",
                alpha=0.5,
                label=f'Position of "{token}" (analyzed)',
            )

        # Set x-axis labels to tokens
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90)

        # Add title and labels
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(
                f"Comparison of explanations for token \"{token}\" in:\n'{self.sentence}'",
                fontsize=14,
            )

        ax.set_ylabel("Normalized Influence Score", fontsize=12)
        ax.set_xlabel("Tokens", fontsize=12)

        # Add a horizontal line at y=0
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # Add legend
        ax.legend(
            title="Explanation Methods", bbox_to_anchor=(1.05, 1), loc="upper left"
        )

        # Adjust layout
        plt.tight_layout()

        # Save if requested
        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")

        # Show if requested
        if show:
            plt.show()

        return plt


    # def plot_self_comparison(
    #     self, save_path=None, show=True, min_influence=0.05, title=None, exponential_scale=1.0
    # ):
    #     import numpy as np
    #     from d3blocks import D3Blocks
    #     import pandas as pd

    #     n_tokens = len(self.tokens)
    #     influence_matrix = np.zeros((n_tokens, n_tokens))

    #     # Build an NxN matrix
    #     for main_pos, main_data in self.scores.items():
    #         for sub_pos, sub_data in main_data["intp"].items():
    #             influence_matrix[main_pos, sub_pos] = abs(sub_data["score"])
                    
    #     # Filter out tiny values
    #     influence_matrix[influence_matrix < min_influence] = 0

    #     non_zero_mask = influence_matrix > 0
    #     if np.any(non_zero_mask):
    #         # Square the values to make large differences more pronounced
    #         influence_matrix[non_zero_mask] = abs(influence_matrix[non_zero_mask]) * exponential_scale

    #     # Create token:position labels to differentiate repeated tokens
    #     token_position_labels = [f"{token}:{pos}" for pos, token in enumerate(self.tokens)]
        
    #     # Convert adjacency matrix to DataFrame with token:position labels
    #     adj_df = pd.DataFrame(
    #         influence_matrix, 
    #         index=token_position_labels, 
    #         columns=token_position_labels
    #     )

    #     # Create an edge list: "source", "target", "weight"
    #     edgelist_df = adj_df.stack().reset_index()
    #     edgelist_df.columns = ["target", "source", "weight"]
    #     edgelist_df = edgelist_df[edgelist_df["weight"] != 0]
        
    #     # Remove self loops (where source == target)
    #     edgelist_df = edgelist_df[edgelist_df["source"] != edgelist_df["target"]]
        
    #     print(edgelist_df)
    #     d3 = D3Blocks()
    #     d3.chord(
    #         edgelist_df,
    #         cmap="rainbow",
    #         title=title if title else f"Token Influence Relationships - {self.explanation_type}",
    #         filepath=save_path if save_path else None,
    #     )

    #     return d3
    

    def plot_self_comparison(
        self, save_path=None, show=True, min_influence=0.05, title=None, exponential_scale=1.0
    ):
        

        import numpy as np
        from d3blocks import D3Blocks
        import pandas as pd
        import os
        import time
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        import matplotlib.pyplot as plt
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM

        n_tokens = len(self.tokens)
        influence_matrix = np.zeros((n_tokens, n_tokens))

        # Build an NxN matrix
        for main_pos, main_data in self.scores.items():
            for sub_pos, sub_data in main_data["intp"].items():
                influence_matrix[main_pos, sub_pos] = abs(sub_data["score"])
                    
        # Filter out tiny values
        influence_matrix[influence_matrix < min_influence] = 0

        non_zero_mask = influence_matrix > 0
        if np.any(non_zero_mask):
            # Square the values to make large differences more pronounced
            influence_matrix[non_zero_mask] = abs(influence_matrix[non_zero_mask]) * exponential_scale

        # Create token:position labels to differentiate repeated tokens
        token_position_labels = [f"{token}:{pos}" for pos, token in enumerate(self.tokens)]
        
        # Convert adjacency matrix to DataFrame with token:position labels
        adj_df = pd.DataFrame(
            influence_matrix, 
            index=token_position_labels, 
            columns=token_position_labels
        )

        # Create an edge list: "source", "target", "weight"
        edgelist_df = adj_df.stack().reset_index()
        edgelist_df.columns = ["target", "source", "weight"]
        edgelist_df = edgelist_df[edgelist_df["weight"] != 0]
        
        # Remove self loops (where source == target)
        edgelist_df = edgelist_df[edgelist_df["source"] != edgelist_df["target"]]

        # Create a temporary HTML file path if save_path is not provided
        html_path = save_path if save_path else "temp_chord_diagram.html"
        
        # Generate SVG path from HTML path
        svg_path = os.path.splitext(html_path)[0] + ".svg"
        
        print(edgelist_df)
        d3 = D3Blocks()
        d3.chord(
            edgelist_df,
            cmap="rainbow",
            title=title if title else f"Token Influence Relationships - {self.explanation_type}",
            filepath=html_path,
        )
        
        # Export to SVG using headless browser
        try:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--disable-gpu')
            browser = webdriver.Chrome(options=options)
            
            # Load the HTML file
            browser.get('file://' + os.path.abspath(html_path))
            
            # Wait for the visualization to render
            time.sleep(2)
            
            # Extract the SVG content
            svg_content = browser.execute_script("""
                var svg = document.querySelector('svg');
                return svg.outerHTML;
            """)
            
            # Save SVG to file
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(svg_content)
                
            browser.quit()
            
            print(f"SVG saved to {svg_path}")
            
            # Display the SVG in matplotlib if show is True
            if show:
                # Convert SVG to a format matplotlib can display
                drawing = svg2rlg(svg_path)
                if drawing is not None:
                    img = renderPM.drawToPIL(drawing)
                    
                    plt.figure(figsize=(10, 10))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(f"Token Influence Relationships - {self.explanation_type}")
                    plt.tight_layout()
                    plt.show()
                
        except Exception as e:
            print(f"Error saving/displaying SVG: {str(e)}")
            # Fall back to showing the HTML
            if show:
                d3.show(filepath=html_path)

        return d3