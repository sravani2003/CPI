import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
import ast
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplcursors
import numpy as np
import math


# Function to parse dictionary strings
def parse_dict_str(dict_str):
    try:
        return ast.literal_eval(dict_str)
    except ValueError as e:
        print(f"Error parsing string: {dict_str}")
        return {}


# Function to calculate similarity score
def calculate_similarity(details1, details2, compulsory_features):
    score = 0
    for key in details1.keys():
        if key in details2 and details1[key] == details2[key]:
            score += 1
    return score


# Function to find similar products
def find_similar_products(asin, price_min, price_max, df, compulsory_features, same_brand_option):
    df['brand'] = df['brand'].astype(str).fillna('')
    target_product = df[df['ASIN'] == asin].iloc[0]
    target_details = {**target_product['Product Details'], **target_product['Glance Icon Details']}
    target_brand = target_product['brand']

    similarities = []
    for index, row in df.iterrows():
            if row['ASIN'] == asin:
                continue
            if same_brand_option == 'only' and str(row['brand']).lower() != target_brand.lower():
                continue
            if same_brand_option == 'omit' and str(row['brand']).lower() == target_brand.lower():
                continue
            if price_min <= row['price'] <= price_max:
                compare_details = {**row['Product Details'], **row['Glance Icon Details']}
                # Check compulsory features
                compulsory_match = all(
                    compare_details.get(feature) == target_details.get(feature) for feature in compulsory_features)
                if compulsory_match:
                    score = calculate_similarity(target_details, compare_details, compulsory_features)
                    similarities.append((row['ASIN'], row['product_title'], row['price'], score, row['Product Details']))


    similarities = sorted(similarities, key=lambda x: x[3], reverse=True)
    return similarities[:100]


# Function to calculate CPI score
def calculate_cpi_score(price, competitor_prices):
    percentile = 100 * (competitor_prices < price).mean()
    cpi_score = 10 - (percentile / 10)
    return cpi_score


# Load data

df = pd.read_csv("C:\\Users\\User\\Downloads\\price_tracker_analysis\\scraped_data_pet.csv", on_bad_lines='skip')
df2 = pd.read_csv("C:\\Users\\User\\Downloads\\price_tracker_analysis\\pet_price.csv")

# Remove duplicate ASINs from df2
df2 = df2.drop_duplicates(subset=['asin'], keep='first')

# Convert ASINs to uppercase
df2['asin'] = df2['asin'].str.upper()
df['ASIN'] = df['ASIN'].str.upper()

# Remove duplicate ASINs from df after conversion to uppercase
df = df.drop_duplicates(subset=['ASIN'], keep='first')

# Merge the DataFrames to ensure no duplicate ASINs across both
combined_asins = pd.concat([df['ASIN'], df2['asin']]).drop_duplicates(keep='first')

# Filter the original DataFrames to include only unique ASINs
df = df[df['ASIN'].isin(combined_asins)]
df2 = df2[df2['asin'].isin(combined_asins)]

df = pd.merge(df, df2[['asin', 'product_title', 'price', 'brand']], left_on='ASIN', right_on='asin', how='left')
df['Product Details'] = df['Product Details'].apply(parse_dict_str)
df['Glance Icon Details'] = df['Glance Icon Details'].apply(parse_dict_str)
df['Option'] = df['Option'].apply(parse_dict_str)

def merge_dicts(row):
    product_details = row['Product Details']
    option_details = row['Option']
    product_details.update(option_details)
    return product_details

# Apply the merge function
df['Product Details'] = df.apply(merge_dicts, axis=1)

print(df['Product Details'][10])


# Format product details
def format_details(details):
    return "\n".join([f"{key}: {value}" for key, value in details.items()])


# First window: Collect user input and show features
def show_features():
    asin = asin_entry.get()
    if asin not in df['ASIN'].values:
        messagebox.showerror("Error", "ASIN not found.")
        return

    target_product = df[df['ASIN'] == asin].iloc[0]
    product_details = {**target_product['Product Details'], **target_product['Glance Icon Details']}

    # Clear previous checkboxes
    for widget in feature_frame.winfo_children():
        widget.destroy()

    tk.Label(feature_frame, text="Select Compulsory Features:", font=("Helvetica", 10, "bold")).pack(anchor='w')

    for feature in product_details.keys():
        var = tk.BooleanVar()
        tk.Checkbutton(feature_frame, text=feature, variable=var).pack(anchor='w')
        compulsory_features_vars[feature] = var

    # Clear previous product details
    for widget in product_details_frame.winfo_children():
        widget.destroy()

    # Show product details
    tk.Label(product_details_frame, text="Product Details:", font=("Helvetica", 10, "bold")).pack(anchor='w')
    details_text = tk.Text(product_details_frame, wrap=tk.WORD, height=10)
    details_text.pack(fill=tk.BOTH, expand=True)
    details_text.insert(tk.END, format_details(product_details))
    details_text.config(state=tk.DISABLED)


# Second window: Display analysis results, including scatter plot and CPI gauge chart
def run_analysis():
    asin = asin_entry.get()
    if asin not in df['ASIN'].values:
        messagebox.showerror("Error", "ASIN not found.")
        return

    try:
        price_min = float(price_min_entry.get())
        price_max = float(price_max_entry.get())
    except ValueError:
        messagebox.showerror("Error", "Invalid price range.")
        return

    same_brand_option = same_brand_var.get()

    compulsory_features = [feature for feature, var in compulsory_features_vars.items() if var.get()]

    similar_products = find_similar_products(asin, price_min, price_max, df, compulsory_features, same_brand_option)

    target_product = df[df['ASIN'] == asin].iloc[0]
    target_price = target_product['price']
    target_title = target_product['product_title']
    target_score = 100  # Maximum score for the target product
    target_details = target_product['Product Details']

    # Create a new window for the analysis results
    result_window = tk.Toplevel(root)
    result_window.title("Analysis Results")

    result_frame = ttk.Frame(result_window, padding=(10, 10))
    result_frame.pack(fill=tk.BOTH, expand=True)

    # Configure grid layout
    result_frame.columnconfigure(0, weight=1)
    result_frame.rowconfigure(0, weight=1)
    result_frame.rowconfigure(1, weight=1)

    # Frame for plot
    plot_frame = ttk.LabelFrame(result_frame, text="Scatter Plot", padding=(10, 10))
    plot_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

    # Plot the results
    fig, ax = plt.subplots(figsize=(12, 8))

    asins = [p[0] for p in similar_products]
    titles = [p[1] for p in similar_products]
    prices = [p[2] for p in similar_products]
    scores = [p[3] for p in similar_products]
    details = [p[4] for p in similar_products]
    indices = range(len(asins))

    scatter = ax.scatter(indices, prices, c=scores, cmap='viridis', s=50, label='Similar Products')
    target_scatter = ax.scatter([-1], [target_price], c='red', marker='*', s=200,
                                label='Target Product')  # Add target product marker

    plt.colorbar(scatter, ax=ax, label='Similarity Score')
    ax.set_xlabel('Index')
    ax.set_ylabel('Price')
    ax.set_title(f'Competitor Analysis for: {target_title}')
    ax.legend()
    ax.grid(True)

    cursor = mplcursors.cursor(scatter, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(
        f"ASIN: {asins[sel.target.index]}\nTitle: {titles[sel.target.index]}\nPrice: ${prices[sel.target.index]:.2f}\nScore: {scores[sel.target.index]:.2f}\nDetails:\n{format_details(details[sel.target.index])}"
    ))

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Calculate and display CPI
    competitor_prices = np.array(prices)
    cpi_score = calculate_cpi_score(target_price, competitor_prices)

    # Frame for CPI gauge chart
    cpi_frame = ttk.LabelFrame(result_frame, text="CPI Score", padding=(10, 10))
    cpi_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=10)

    fig_cpi, ax_cpi = plt.subplots(figsize=(6, 3), subplot_kw={'projection': 'polar'})
    ax_cpi.set_theta_offset(math.pi)
    ax_cpi.set_theta_direction(-1)

    # Draw the CPI gauge
    categories = [''] * 10
    angles = np.linspace(0, np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    values = [0] * 10
    values += values[:1]

    ax_cpi.fill(angles, values, color='grey', alpha=0.25)
    ax_cpi.fill(angles, values, color='grey', alpha=0.5)

    score_angle = (cpi_score / 10) * np.pi
    ax_cpi.plot([0, score_angle], [0, 10], color='blue', linewidth=2, linestyle='solid')
    ax_cpi.fill([0, score_angle, score_angle, 0], [0, 10, 0, 0], color='blue', alpha=0.5)

    ax_cpi.set_ylim(0, 10)
    ax_cpi.set_yticklabels([])
    ax_cpi.set_xticklabels([])

    # Display CPI value
    ax_cpi.text(0, 0, f'{cpi_score:.2f}', horizontalalignment='center', verticalalignment='center', fontsize=20,
                fontweight='bold')

    canvas_cpi = FigureCanvasTkAgg(fig_cpi, master=cpi_frame)
    canvas_cpi.draw()
    canvas_cpi.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


# Initialize Tkinter window
root = tk.Tk()
root.title("ASIN Competitor Analysis")

# Main frame for input and plot
main_frame = ttk.Frame(root, padding=(10, 10))
main_frame.pack(fill=tk.BOTH, expand=True)

# Configure grid layout
main_frame.columnconfigure(0, weight=1)
main_frame.columnconfigure(1, weight=4)
main_frame.rowconfigure(0, weight=1)

# Create input frame
input_frame = ttk.LabelFrame(main_frame, text="Input Parameters", padding=(10, 10))
input_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

ttk.Label(input_frame, text="ASIN:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
asin_entry = ttk.Entry(input_frame)
asin_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')

ttk.Label(input_frame, text="Price Min:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
price_min_entry = ttk.Entry(input_frame)
price_min_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')

ttk.Label(input_frame, text="Price Max:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
price_max_entry = ttk.Entry(input_frame)
price_max_entry.grid(row=2, column=1, padx=5, pady=5, sticky='w')

same_brand_var = tk.StringVar(value='all')
ttk.Radiobutton(input_frame, text="Include all brands", variable=same_brand_var, value='all').grid(row=3, column=0,
                                                                                                   columnspan=2, pady=5)
ttk.Radiobutton(input_frame, text="Show only same brand products", variable=same_brand_var, value='only').grid(row=4,
                                                                                                               column=0,
                                                                                                               columnspan=2,
                                                                                                               pady=5)
ttk.Radiobutton(input_frame, text="Omit same brand products", variable=same_brand_var, value='omit').grid(row=5,
                                                                                                          column=0,
                                                                                                          columnspan=2,
                                                                                                          pady=5)

# Button to show features
show_features_button = ttk.Button(input_frame, text="Show Features", command=show_features)
show_features_button.grid(row=6, column=0, columnspan=2, pady=10)

# Frame for features with scrollbar
feature_frame_container = ttk.Frame(input_frame)
feature_frame_container.grid(row=7, column=0, columnspan=2, pady=10)

feature_canvas = tk.Canvas(feature_frame_container, height=200)
feature_frame = ttk.Frame(feature_canvas)
scrollbar = ttk.Scrollbar(feature_frame_container, orient="vertical", command=feature_canvas.yview)
feature_canvas.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side="right", fill="y")
feature_canvas.pack(side="left", fill="both", expand=True)
feature_canvas.create_window((0, 0), window=feature_frame, anchor='nw')


def onFrameConfigure(canvas):
    canvas.configure(scrollregion=canvas.bbox("all"))


feature_frame.bind("<Configure>", lambda event, canvas=feature_canvas: onFrameConfigure(canvas))

# Frame for product details
product_details_frame = ttk.LabelFrame(input_frame, text="Product Details", padding=(10, 10))
product_details_frame.grid(row=8, column=0, columnspan=2, pady=10, sticky='nsew')

# Button to run analysis
analyze_button = ttk.Button(input_frame, text="Analyze", command=run_analysis)
analyze_button.grid(row=9, column=0, columnspan=2, pady=10)

compulsory_features_vars = {}

root.mainloop()
