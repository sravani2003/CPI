import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import ast
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplcursors
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import Levenshtein

# Ensure nltk tokenization resources are downloaded
nltk.download('punkt')

# Tokenization function
def tokenize(text):
    tokens = word_tokenize(text.lower())
    return set(tokens)

# Jaccard similarity function
def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

# Combined Jaccard-Levenshtein similarity
def title_similarity(title1, title2):
    tokens1 = tokenize(title1)
    tokens2 = tokenize(title2)
    jaccard_score = jaccard_similarity(tokens1, tokens2)
    levenshtein_score = Levenshtein.distance(title1, title2)
    max_len = max(len(title1), len(title2))
    levenshtein_similarity = (max_len - levenshtein_score) / max_len
    return (jaccard_score + levenshtein_similarity) / 2 * 100

# Function to parse dictionary strings
def parse_dict_str(dict_str):
    try:
        return ast.literal_eval(dict_str)
    except ValueError as e:
        print(f"Error parsing string: {dict_str}")
        return {}
    
# Function to calculate similarity score
def calculate_similarity(details1, details2, title1, title2):
    score = 0
    total_keys = len(details1.keys())
    for key in details1.keys():
        if key in details2:
            if key == 'Material':
                materials1 = set(details1[key].split(', '))
                materials2 = set(details2[key].split(', '))
                score += jaccard_similarity(materials1, materials2)
            elif details1[key] == details2[key]:
                score += 1
    if total_keys > 0:
        details_score = (score / total_keys) * 100  # Normalize to 0-100 scale
    else:
        details_score = 0
    title_score = title_similarity(title1, title2)
    return details_score, title_score

# Function to calculate weighted score
def calculate_weighted_score(details_score, title_score):
    weighted_score = 0.6 * details_score + 0.4 * title_score
    return weighted_score

# Function to find similar products
def find_similar_products(asin, price_min, price_max, df, compulsory_features, same_brand_option):
    df['brand'] = df['brand'].astype(str).fillna('')
    target_product = df[df['ASIN'] == asin].iloc[0]
    target_details = {**target_product['Product Details'], **target_product['Glance Icon Details']}
    target_brand = target_product['brand']
    target_title = target_product['product_title']

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
            compare_title = row['product_title']
            compulsory_match = all(
                compare_details.get(feature) == target_details.get(feature) for feature in compulsory_features)
            if compulsory_match:
                details_score, title_score = calculate_similarity(
                    target_details, compare_details, target_title, compare_title)
                weighted_score = calculate_weighted_score(details_score, title_score)
                product_details = row['Product Details']
                similarities.append((row['ASIN'], row['product_title'], row['price'], details_score, title_score, weighted_score, product_details))

    similarities = sorted(similarities, key=lambda x: x[5], reverse=True)
    return similarities[:100]

# Function to calculate CPI score
def calculate_cpi_score(price, competitor_scores):
    percentile = 100 * (competitor_scores < price).mean()
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

# Apply the merge function
df = pd.merge(df, df2[['asin', 'product_title', 'price', 'brand']], left_on='ASIN', right_on='asin', how='inner')

# Format product details
df['Product Details'] = df['Product Details'].apply(parse_dict_str)
df['Glance Icon Details'] = df['Glance Icon Details'].apply(parse_dict_str)

# Function to format details for display
def format_details(details):
    return "\n".join(f"{k}: {v}" for k, v in details.items())

# Function to show features in a scrollable frame
def show_features():
    product_asin = asin_entry.get()
    if product_asin not in df['ASIN'].values:
        messagebox.showerror("Error", "ASIN not found.")
        return
    
    product_row = df[df['ASIN'] == product_asin].iloc[0]
    product_details = {**product_row['Product Details'], **product_row['Glance Icon Details']}
    for widget in feature_frame.winfo_children():
        widget.destroy()
    
    compulsory_features_vars.clear()
    
    for feature, value in product_details.items():
        var = tk.BooleanVar()
        chk = ttk.Checkbutton(feature_frame, text=f"{feature}: {value}", variable=var)
        chk.pack(anchor='w')
        compulsory_features_vars[feature] = var

# Function to run analysis and display results
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

    print("Similar Products: ", similar_products)

    target_product = df[df['ASIN'] == asin].iloc[0]
    target_price = target_product['price']
    target_title = target_product['product_title']
    target_score = 100  # Maximum score for the target product
    target_details = target_product['Product Details']

    result_window = tk.Toplevel(root)
    result_window.title("Analysis Results")
    result_frame = ttk.Frame(result_window, padding=(10, 10))
    result_frame.pack(fill=tk.BOTH, expand=True)
    result_frame.columnconfigure(0, weight=1)
    result_frame.rowconfigure(0, weight=1)
    result_frame.rowconfigure(1, weight=1)
    
    plot_frame = ttk.LabelFrame(result_frame, text="Scatter Plot", padding=(10, 10))
    plot_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()

    asins = [p[0] for p in similar_products]
    titles = [p[1] for p in similar_products]
    prices = [p[2] for p in similar_products]
    details_scores = [p[3] for p in similar_products]
    title_scores = [p[4] for p in similar_products]
    weighted_scores = [p[5] for p in similar_products]
    product_details = [p[6] for p in similar_products]

    indices = np.arange(len(similar_products))
    fig, ax1 = plt.subplots()

    sc = ax1.scatter(indices, prices, c='blue', s=100, alpha=0.6, edgecolors='w', label='Price')
    sc2 = ax1.scatter(indices, weighted_scores, c=weighted_scores, cmap='cool', s=100, alpha=0.6, edgecolors='k', label='Weighted Score')
    ax1.scatter(target_price, target_score, c='red', label='Target Product', marker='*', s=200, edgecolors='black')

    ax1.set_xlabel('Product Indices (Descending Order by Score)')
    ax1.set_ylabel('Price ($)')
    ax2.set_ylabel('Weighted Score')
    ax1.invert_xaxis()
    product_name = df.loc[df['ASIN'] == asin, 'product_name'].values[0]
    ax1.set_title(f'Similar Products for: {product_name}')

    fig.colorbar(sc2, ax=ax2, label='Weighted Score')
    ax1.legend()
    ax1.grid(True)

    cursor = mplcursors.cursor(sc, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(
    f"ASIN: {asins[sel.target.index]}\nTitle: {titles[sel.target.index]}\n"
    f"Price: ${prices[sel.target.index]:.2f}\n"
    f"Product Details:\n" +
    "\n".join([f"{key}: {value}" for key, value in product_details[sel.target.index].items()])
    ))

    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    details_frame = ttk.LabelFrame(result_frame, text="Product Details", padding=(10, 10))
    details_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=10)

    product_text = tk.Text(details_frame, wrap=tk.WORD, height=10)
    product_text.insert(tk.END, f"ASIN: {asin}\nTitle: {target_title}\nPrice: ${target_price:.2f}\nDetails:\n")
    product_text.insert(tk.END, format_details(target_details))
    product_text.config(state=tk.DISABLED)
    product_text.pack(fill=tk.BOTH, expand=True)

    cpi_frame = ttk.LabelFrame(result_frame, text="CPI Score", padding=(10, 10))
    cpi_frame.grid(row=2, column=0, sticky='nsew', padx=10, pady=10)

    competitor_prices = np.array(prices)
    cpi_score = calculate_cpi_score(target_price, competitor_prices)

    fig_cpi, ax_cpi = plt.subplots(figsize=(4, 4), subplot_kw=dict(aspect="equal"))
    ax_cpi.pie([cpi_score, 10 - cpi_score], labels=['CPI Score', 'Remaining'], colors=['blue', 'grey'], startangle=90, counterclock=False)
    ax_cpi.text(0, 0, f'{cpi_score:.2f}', horizontalalignment='center', verticalalignment='center', fontsize=20,
                fontweight='bold')

    canvas_cpi = FigureCanvasTkAgg(fig_cpi, master=cpi_frame)
    canvas_cpi.draw()
    canvas_cpi.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Initialize Tkinter Window
root = tk.Tk()
root.title("ASIN Competitor Analysis")

main_frame = ttk.Frame(root, padding=(10, 10))
main_frame.pack(fill=tk.BOTH, expand=True)

main_frame.columnconfigure(0, weight=1)
main_frame.columnconfigure(1, weight=4)
main_frame.rowconfigure(0, weight=1)

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
ttk.Radiobutton(input_frame, text="Include all brands", variable=same_brand_var, value='all').grid(row=3, column=0, columnspan=2, pady=5)
ttk.Radiobutton(input_frame, text="Show only same brand products", variable=same_brand_var, value='only').grid(row=4, column=0, columnspan=2, pady=5)
ttk.Radiobutton(input_frame, text="Omit same brand products", variable=same_brand_var, value='omit').grid(row=5, column=0, columnspan=2, pady=5)

show_features_button = ttk.Button(input_frame, text="Show Features", command=show_features)
show_features_button.grid(row=6, column=0, columnspan=2, pady=10)

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

product_details_frame = ttk.LabelFrame(input_frame, text="Product Details", padding=(10, 10))
product_details_frame.grid(row=8, column=0, columnspan=2, pady=10, sticky='nsew')

analyze_button = ttk.Button(input_frame, text="Analyze", command=run_analysis)
analyze_button.grid(row=9, column=0, columnspan=2, pady=10)

compulsory_features_vars = {}

root.mainloop()
