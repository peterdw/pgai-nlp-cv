import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import os
import json

from utils import read_csv_file

df = pd.read_feather("data/df_emails_cleaned.feather")

# df_emails = read_csv_file("data/metadata.csv")
# print(df.columns)
# print('META_SuppliedEmail' in df.columns)

class TextComparerApp:
    def __init__(self, root, dataframe):
        self.root = root
        self.df = dataframe
        self.index = 0
        self.total = len(dataframe)
        self.history_file = "search_history.json"
        self.search_history = self.load_search_history()
        self.scroll_job = None
        self.scroll_delay = 250  # milliseconds between scroll steps
        self.scroll_mode = None  # 'next' or 'prev'

        self.root.title("Text Comparer")
        self.root.geometry("1000x700")

        self.filtered_df = self.df.copy()
        self.regex_enabled = tk.BooleanVar()

        # Root grid layout
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Build UI
        self.build_top()
        self.build_text_areas()
        self.build_navigator()

        self.load_record()

    def on_search_selected(self, event=None):
        selected_text = self.search_var.get()
        for entry in self.search_history:
            if entry['text'] == selected_text:
                self.regex_enabled.set(entry.get('regex', False))
                break

    def load_search_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Could not load search history: {e}")
        return []

    def save_search_history(self):
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.search_history, f, indent=2)
        except Exception as e:
            print(f"Could not save search history: {e}")

    def update_dropdown_counts(self):
        # Recalculate dropdown items based on the current filtered_df
        current_df = self.filtered_df

        # --- Update Complaint Type Dropdown ---
        complaint_counts = current_df["AGR_Type_of_Complaint__c"].value_counts()
        has_missing_complaints = current_df["AGR_Type_of_Complaint__c"].isna().any()

        complaint_dropdown_items = ["<All>"]
        if has_missing_complaints:
            missing_count = current_df["AGR_Type_of_Complaint__c"].isna().sum()
            complaint_dropdown_items.append(f"<Missing> ({missing_count})")

        self.complaint_display_to_value.clear()
        for complaint_type, count in complaint_counts.items():
            display = f"{complaint_type} ({count})"
            complaint_dropdown_items.append(display)
            self.complaint_display_to_value[display] = complaint_type

        current_complaint = self.complaint_filter_box.get()
        self.complaint_filter_box['values'] = complaint_dropdown_items
        if current_complaint in complaint_dropdown_items:
            self.complaint_filter_box.set(current_complaint)
        else:
            self.complaint_filter_box.set("<All>")

        # --- Update Language Dropdown ---
        lang_df = current_df[["LangName", "Language"]].copy()
        has_missing_lang = lang_df["Language"].isna().any()

        lang_df_clean = lang_df.dropna(subset=["LangName"])
        lang_counts = lang_df_clean["LangName"].value_counts()
        lang_rows = lang_df_clean.drop_duplicates().set_index("LangName")["Language"]

        lang_dropdown_items = ["<All>"]
        if has_missing_lang:
            lang_dropdown_items.append("<Missing>")

        self.lang_display_to_code.clear()
        for lang_name, count in lang_counts.items():
            lang_code = lang_rows.get(lang_name)
            display = f"{lang_name} ({count})"
            lang_dropdown_items.append(display)
            self.lang_display_to_code[display] = lang_code

        current_lang = self.language_filter_box.get()
        self.language_filter_box['values'] = lang_dropdown_items
        if current_lang in lang_dropdown_items:
            self.language_filter_box.set(current_lang)
        else:
            self.language_filter_box.set("<All>")

        # --- Update Priority Dropdown ---
        priority_counts = current_df["Priority"].value_counts(dropna=False)
        priority_dropdown_items = ["<All>"]
        self.priority_display_to_value.clear()

        for value, count in priority_counts.items():
            display = "<Missing>" if pd.isna(value) else f"{value} ({count})"
            priority_dropdown_items.append(display)
            self.priority_display_to_value[display] = value

        current_priority = self.priority_filter_box.get()
        self.priority_filter_box['values'] = priority_dropdown_items
        if current_priority in priority_dropdown_items:
            self.priority_filter_box.set(current_priority)
        else:
            self.priority_filter_box.set("<All>")

    def build_top(self):
        self.top_frame = ttk.Frame(self.root, padding=10)
        self.top_frame.grid(row=0, column=0, sticky="ew")
        self.top_frame.columnconfigure(1, weight=1)

        shared_width = 40

        # === Row 0: Search Label and Row ===
        ttk.Label(self.top_frame, text="Search in TextBody:").grid(row=0, column=0, sticky="w", padx=(0, 10))

        search_row = ttk.Frame(self.top_frame)
        search_row.grid(row=0, column=1, sticky="w")

        self.search_var = tk.StringVar()
        self.search_box = ttk.Combobox(
            search_row,
            textvariable=self.search_var,
            width=40,
            values=[entry['text'] for entry in self.search_history]
        )
        self.search_box.pack(side="left")
        self.search_box.bind("<Return>", lambda e: self.apply_filter())
        self.search_box.bind("<<ComboboxSelected>>", self.on_search_selected)

        self.regex_enabled = tk.BooleanVar()
        ttk.Checkbutton(search_row, text="Use Regex", variable=self.regex_enabled).pack(side="left", padx=5)
        ttk.Button(search_row, text="🔍 Search", command=self.apply_filter).pack(side="left", padx=5)
        ttk.Button(search_row, text="Clear Search", command=self.clear_filter).pack(side="left", padx=5)

        # === Row 0b: Character Limit ===
        self.char_limit_var = tk.StringVar()
        char_limit_row = ttk.Frame(self.top_frame)
        char_limit_row.grid(row=0, column=2, sticky="w", padx=(10, 0))

        ttk.Label(char_limit_row, text="Max Length:").pack(side="left", padx=(0, 5))
        self.char_limit_entry = ttk.Entry(char_limit_row, textvariable=self.char_limit_var, width=8)
        self.char_limit_entry.pack(side="left")
        self.char_limit_entry.bind("<Return>", lambda e: self.apply_filter())

        # === Row 1: Complaint Type ===
        self.complaint_filter_var = tk.StringVar()
        self.complaint_display_to_value = {}
        complaint_counts = self.df["AGR_Type_of_Complaint__c"].value_counts()
        complaint_values = ["<All>"]
        if self.df["AGR_Type_of_Complaint__c"].isna().any():
            count = self.df["AGR_Type_of_Complaint__c"].isna().sum()
            complaint_values.append(f"<Missing> ({count})")
        for k, v in complaint_counts.items():
            label = f"{k} ({v})"
            complaint_values.append(label)
            self.complaint_display_to_value[label] = k
        ttk.Label(self.top_frame, text="Complaint Type:").grid(row=1, column=0, sticky="w", pady=4)
        self.complaint_filter_box = ttk.Combobox(
            self.top_frame,
            textvariable=self.complaint_filter_var,
            values=complaint_values,
            state="readonly",
            width=shared_width
        )
        self.complaint_filter_box.grid(row=1, column=1, sticky="w", pady=4)
        self.complaint_filter_box.set("<All>")
        self.complaint_filter_box.bind("<<ComboboxSelected>>", lambda e: self.apply_filter())

        # === Row 2: Language ===
        self.language_filter_var = tk.StringVar()
        self.lang_display_to_code = {}

        # Extract distinct values
        distinct_language = df["Language"].dropna().drop_duplicates().reset_index(drop=True)
        distinct_langname = df["LangName"].dropna().drop_duplicates().reset_index(drop=True)

        # Save to CSV
        distinct_language.to_csv("data/distinct_language_values.csv", index=False, header=["Language"])
        distinct_langname.to_csv("data/distinct_langname_values.csv", index=False, header=["LangName"])

        lang_df = self.df[["LangName", "Language"]].dropna(subset=["LangName"])
        lang_counts = lang_df["LangName"].value_counts()
        lang_map = lang_df.drop_duplicates(subset=["LangName"]).set_index("LangName")["Language"]
        lang_values = ["<All>"]
        if self.df["Language"].isna().any():
            lang_values.append("<Missing>")
        for name, count in lang_counts.items():
            label = f"{name} ({count})"
            lang_values.append(label)
            self.lang_display_to_code[label] = lang_map.get(name)
        ttk.Label(self.top_frame, text="Language:").grid(row=2, column=0, sticky="w", pady=4)
        self.language_filter_box = ttk.Combobox(
            self.top_frame,
            textvariable=self.language_filter_var,
            values=lang_values,
            state="readonly",
            width=shared_width
        )
        self.language_filter_box.grid(row=2, column=1, sticky="w", pady=4)
        self.language_filter_box.set("<All>")
        self.language_filter_box.bind("<<ComboboxSelected>>", lambda e: self.apply_filter())

        # === Row 3: Priority ===
        self.priority_filter_var = tk.StringVar()
        self.priority_display_to_value = {}
        priority_counts = self.df["Priority"].value_counts(dropna=False)
        priority_values = ["<All>"]
        for val, count in priority_counts.items():
            label = f"<Missing> ({count})" if pd.isna(val) else f"{val} ({count})"
            priority_values.append(label)
            self.priority_display_to_value[label] = val
        ttk.Label(self.top_frame, text="Priority:").grid(row=3, column=0, sticky="w", pady=4)
        self.priority_filter_box = ttk.Combobox(
            self.top_frame,
            textvariable=self.priority_filter_var,
            values=priority_values,
            state="readonly",
            width=shared_width
        )
        self.priority_filter_box.grid(row=3, column=1, sticky="w", pady=4)
        self.priority_filter_box.set("<All>")
        self.priority_filter_box.bind("<<ComboboxSelected>>", lambda e: self.apply_filter())

    def highlight_matches(self, pattern, text_widget, use_regex=False):
        import re
        text_widget.tag_remove("highlight", "1.0", tk.END)  # Clear old highlights
        text_widget.tag_configure("highlight", background="yellow", foreground="black")

        content = text_widget.get("1.0", tk.END)
        if not pattern.strip():
            return

        try:
            if use_regex:
                matches = re.finditer(pattern, content, re.IGNORECASE)
            else:
                matches = [(m.start(), m.end()) for m in re.finditer(re.escape(pattern), content, re.IGNORECASE)]

            for match in matches:
                if use_regex:
                    start, end = match.start(), match.end()
                else:
                    start, end = match

                start_idx = f"1.0 + {start} chars"
                end_idx = f"1.0 + {end} chars"
                text_widget.tag_add("highlight", start_idx, end_idx)

        except re.error:
            messagebox.showerror("Regex Error", "Invalid regular expression.")

    def build_text_areas(self):
        # === Paned Window ===
        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        # === Left Pane ===
        left_pane = ttk.Frame(paned, width=250)
        paned.add(left_pane, weight=0)  # non-expanding

        ttk.Label(left_pane, text="Record ID:").pack(anchor="w", padx=5, pady=(10, 2))
        self.index_id_var = tk.StringVar()
        self.index_id_entry = ttk.Entry(left_pane, textvariable=self.index_id_var, width=30, state='readonly')
        self.index_id_entry.pack(fill="x", padx=5, pady=(0, 10))

        ttk.Label(left_pane, text="Subject:").pack(anchor="w", padx=5, pady=(10, 2))
        self.subject_var = tk.StringVar()
        self.subject_entry = ttk.Entry(left_pane, textvariable=self.subject_var, width=30, state='readonly')
        self.subject_entry.pack(fill="x", padx=5, pady=(0, 10))

        ttk.Label(left_pane, text="Supplied Email:").pack(anchor="w", padx=5, pady=(0, 2))
        self.supplied_email_var = tk.StringVar()
        self.supplied_email_entry = ttk.Entry(left_pane, textvariable=self.supplied_email_var, width=30, state='readonly')
        self.supplied_email_entry.pack(fill="x", padx=5)

        ttk.Label(left_pane, text="Language:").pack(anchor="w", padx=5, pady=(0, 2))
        self.language_var = tk.StringVar()
        self.language_entry = ttk.Entry(left_pane, textvariable=self.language_var, width=30, state='readonly')
        self.language_entry.pack(fill="x", padx=5)

        ttk.Label(left_pane, text="Complaint Type:").pack(anchor="w", padx=5, pady=(10, 2))
        self.complaint_type_var = tk.StringVar()
        self.complaint_type_entry = ttk.Entry(left_pane, textvariable=self.complaint_type_var, width=30,
                                              state='readonly')
        self.complaint_type_entry.pack(fill="x", padx=5)

        ttk.Label(left_pane, text="Complaint:").pack(anchor="w", padx=5, pady=(10, 2))
        self.complaint_var = tk.StringVar()
        self.complaint_entry = ttk.Entry(left_pane, textvariable=self.complaint_var, width=30, state='readonly')
        self.complaint_entry.pack(fill="x", padx=5)

        ttk.Label(left_pane, text="Reason:").pack(anchor="w", padx=5, pady=(10, 2))
        self.reason_var = tk.StringVar()
        self.reason_entry = ttk.Entry(left_pane, textvariable=self.reason_var, width=30, state='readonly')
        self.reason_entry.pack(fill="x", padx=5)

        # === Right Pane with Vertical PanedWindow for Text Areas ===
        right_pane = ttk.Frame(paned)
        right_pane.columnconfigure(0, weight=1)
        right_pane.rowconfigure(0, weight=1)
        paned.add(right_pane, weight=1)

        text_paned = ttk.PanedWindow(right_pane, orient="vertical")
        text_paned.grid(row=0, column=0, sticky="nsew")

        # -- Original Mail Section --
        orig_frame = ttk.Frame(text_paned)
        ttk.Label(orig_frame, text="Original Mail").pack(anchor="w", padx=5, pady=(2, 2))

        # Wrapper for text + vertical scrollbar
        orig_text_frame = ttk.Frame(orig_frame)
        orig_text_frame.pack(fill="both", expand=True)

        self.text_body = tk.Text(orig_text_frame, wrap="none")
        self.text_body.pack(side="left", fill="both", expand=True)

        yscroll1 = ttk.Scrollbar(orig_text_frame, orient="vertical", command=self.text_body.yview)
        yscroll1.pack(side="right", fill="y")
        self.text_body.config(yscrollcommand=yscroll1.set)

        # Place horizontal scrollbar BELOW the text+vscroll frame
        xscroll1 = ttk.Scrollbar(orig_frame, orient="horizontal", command=self.text_body.xview)
        xscroll1.pack(fill="x")
        self.text_body.config(xscrollcommand=xscroll1.set)

        text_paned.add(orig_frame, weight=1)
        self.root.after(50, lambda: self.text_body.event_generate("<Configure>"))

        # # -- Tagged Mail Section --
        # tagged_frame = ttk.Frame(text_paned)
        # ttk.Label(tagged_frame, text="Tagged Mail").pack(anchor="w", padx=5, pady=(2, 2))
        #
        # # Wrapper for text + vertical scrollbar
        # tagged_text_frame = ttk.Frame(tagged_frame)
        # tagged_text_frame.pack(fill="both", expand=True)
        #
        # self.tagged_text_body = tk.Text(tagged_text_frame, wrap="none")
        # self.tagged_text_body.pack(side="left", fill="both", expand=True)
        #
        # yscroll_tagged = ttk.Scrollbar(tagged_text_frame, orient="vertical", command=self.tagged_text_body.yview)
        # yscroll_tagged.pack(side="right", fill="y")
        # self.tagged_text_body.config(yscrollcommand=yscroll_tagged.set)
        #
        # # <-- Horizontal scrollbar now in tagged_text_frame -->
        # xscroll_tagged = ttk.Scrollbar(tagged_frame, orient="horizontal", command=self.tagged_text_body.xview)
        # xscroll_tagged.pack(fill="x")
        # self.tagged_text_body.config(xscrollcommand=xscroll_tagged.set)
        #
        # text_paned.add(tagged_frame, weight=1)
        # self.root.after(50, lambda: self.tagged_text_body.event_generate("<Configure>"))

        # -- Cleaned Mail Section --
        proc_frame = ttk.Frame(text_paned)
        ttk.Label(proc_frame, text="Cleaned Mail").pack(anchor="w", padx=5, pady=(2, 2))

        # Wrapper for text + vertical scrollbar
        proc_text_frame = ttk.Frame(proc_frame)
        proc_text_frame.pack(fill="both", expand=True)

        self.processed_text_body = tk.Text(proc_text_frame, wrap="none")
        self.processed_text_body.pack(side="left", fill="both", expand=True)

        yscroll2 = ttk.Scrollbar(proc_text_frame, orient="vertical", command=self.processed_text_body.yview)
        yscroll2.pack(side="right", fill="y")
        self.processed_text_body.config(yscrollcommand=yscroll2.set)

        # <-- Horizontal scrollbar now in proc_text_frame -->
        xscroll2 = ttk.Scrollbar(proc_frame, orient="horizontal", command=self.processed_text_body.xview)
        xscroll2.pack(fill="x")
        self.processed_text_body.config(xscrollcommand=xscroll2.set)

        text_paned.add(proc_frame, weight=1)
        self.root.after(50, lambda: self.processed_text_body.event_generate("<Configure>"))

    def build_navigator(self):
        # === Bottom frame ===
        nav_frame = ttk.Frame(self.root, padding=(10, 5))
        nav_frame.grid(row=2, column=0, sticky="ew")
        nav_frame.columnconfigure(0, weight=0)

        # === Info label (row 0) ===
        self.info_label_style = ttk.Style()
        self.info_label_style.configure("Info.TLabel", font=("Segoe UI", 10))
        self.info_label_style.configure("Filtered.TLabel", font=("Segoe UI", 10, "bold"), foreground="darkblue")
        self.info_label = ttk.Label(nav_frame, text="", anchor="w", style="Info.TLabel")
        self.info_label.grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 5))

        # === Navigation controls (row 1) ===
        nav_controls = ttk.Frame(nav_frame)
        nav_controls.grid(row=1, column=0, sticky="w")

        self.first_btn = ttk.Button(nav_controls, text="|<", width=4, command=self.go_first)
        self.first_btn.grid(row=0, column=0, padx=(0, 2))

        self.prev_btn = tk.Button(nav_controls, text="<", width=4)
        self.prev_btn.grid(row=0, column=1, padx=2)
        self.prev_btn.bind("<ButtonPress-1>", lambda e: self.start_scrolling("prev"))
        self.prev_btn.bind("<ButtonRelease-1>", lambda e: self.stop_scrolling())

        self.record_entry = ttk.Entry(nav_controls, width=6, justify='center')
        self.record_entry.grid(row=0, column=2, padx=5)
        self.record_entry.bind("<Return>", self.go_to_record)

        self.next_btn = tk.Button(nav_controls, text=">", width=4)
        self.next_btn.grid(row=0, column=3, padx=2)
        self.next_btn.bind("<ButtonPress-1>", lambda e: self.start_scrolling("next"))
        self.next_btn.bind("<ButtonRelease-1>", lambda e: self.stop_scrolling())

        self.last_btn = ttk.Button(nav_controls, text=">|", width=4, command=self.go_last)
        self.last_btn.grid(row=0, column=4, padx=(2, 0))

    def load_record(self):
        self.text_body.delete('1.0', tk.END)
        self.processed_text_body.delete('1.0', tk.END)
        # self.tagged_text_body.delete('1.0', tk.END)

        row = self.filtered_df.iloc[self.index]
        self.index_id_var.set(str(self.filtered_df.index[self.index] + 1))

        self.text_body.insert(tk.END, str(row['TextBody']))
        self.processed_text_body.insert(tk.END, str(row['ProcessedTextBody']))
        # self.tagged_text_body.insert(tk.END, str(row.get('TaggedTextBody', '')))

        # Set subject and language from the row (fallback to empty string if missing)
        self.subject_var.set(str(row.get('Subject', '')))
        self.supplied_email_var.set(str(row.get('META_SuppliedEmail', '')))
        self.language_var.set(str(row.get('Language', '')))
        self.complaint_type_var.set(str(row.get("AGR_Type_of_Complaint__c", "")))
        self.reason_var.set(str(row.get("AGR_Reason_for_Complaint__c", "")))
        self.complaint_var.set(str(row.get("Complaint", "")))

        self.update_info_label()

        # Ensure highlights are updated based on current search
        self.text_body.tag_remove("highlight", "1.0", tk.END)
        query = self.search_var.get().strip()
        if query:
            self.root.after(10,
                            lambda: self.highlight_matches(query, self.text_body, use_regex=self.regex_enabled.get()))

        self.record_entry.delete(0, tk.END)
        self.record_entry.insert(0, str(self.index + 1))

        self.first_btn.config(state=tk.NORMAL if self.index > 0 else tk.DISABLED)
        self.prev_btn.config(state=tk.NORMAL if self.index > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.index < len(self.filtered_df) - 1 else tk.DISABLED)
        self.last_btn.config(state=tk.NORMAL if self.index < len(self.filtered_df) - 1 else tk.DISABLED)

    def next_record(self):
        if self.index < len(self.filtered_df) - 1:
            self.index += 1
            self.load_record()

    def prev_record(self):
        if self.index > 0:
            self.index -= 1
            self.load_record()

    def go_first(self):
        self.index = 0
        self.load_record()

    def go_last(self):
        self.index = len(self.filtered_df) - 1
        self.load_record()

    def go_to_record(self, event=None):
        try:
            record_num = int(self.record_entry.get())
            if 1 <= record_num <= len(self.filtered_df):
                self.index = record_num - 1
                self.load_record()
            else:
                messagebox.showwarning("Invalid Record", f"Enter a number between 1 and {len(self.filtered_df)}")
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid integer.")

    def apply_filter(self):
        query = self.search_var.get().strip()
        self.filtered_df = self.df.copy()

        # === Text Search ===
        if query:
            import re
            if self.regex_enabled.get():
                try:
                    pattern = re.compile(query, re.IGNORECASE)
                    self.filtered_df = self.filtered_df[
                        self.filtered_df["TextBody"].apply(lambda x: bool(pattern.search(str(x))))]
                except re.error:
                    messagebox.showerror("Regex Error", "Invalid regular expression.")
                    return
            else:
                self.filtered_df = self.filtered_df[
                    self.filtered_df["TextBody"].str.contains(query, case=False, na=False)]

        # === Character Length Filter ===
        char_limit = self.char_limit_var.get().strip()
        if char_limit:
            try:
                max_length = int(char_limit)
                self.filtered_df = self.filtered_df[
                    self.filtered_df["ProcessedTextBody"].str.len() <= max_length
                    ]
            except ValueError:
                messagebox.showwarning("Invalid Input", "Character limit must be an integer.")
                return

        # === Complaint Type Filter ===
        selected_complaint_display = self.complaint_filter_var.get().strip()
        if selected_complaint_display.startswith("<Missing>"):
            self.filtered_df = self.filtered_df[self.filtered_df["AGR_Type_of_Complaint__c"].isna()]
        elif selected_complaint_display != "<All>":
            complaint_value = self.complaint_display_to_value.get(selected_complaint_display)
            self.filtered_df = self.filtered_df[
                self.filtered_df["AGR_Type_of_Complaint__c"] == complaint_value
                ]

        # === Language Filter ===
        selected_lang_display = self.language_filter_var.get().strip()
        if selected_lang_display == "<Missing>":
            self.filtered_df = self.filtered_df[self.filtered_df["Language"].isna()]
        elif selected_lang_display != "<All>":
            lang_code = self.lang_display_to_code.get(selected_lang_display)

            if lang_code is None and "Unknown" in selected_lang_display:
                # Filter on LangName (how the dropdown counts it)
                self.filtered_df = self.filtered_df[self.filtered_df["LangName"] == "Unknown"]
            else:
                self.filtered_df = self.filtered_df[self.filtered_df["Language"] == lang_code]

        # === Priority Filter ===
        selected_priority_display = self.priority_filter_var.get().strip()
        if selected_priority_display != "<All>":
            priority_value = self.priority_display_to_value.get(selected_priority_display)
            if pd.isna(priority_value):
                self.filtered_df = self.filtered_df[self.filtered_df["Priority"].isna()]
            else:
                self.filtered_df = self.filtered_df[self.filtered_df["Priority"] == priority_value]

        # === Check Result ===
        if self.filtered_df.empty:
            messagebox.showinfo("No Matches", "No records found.")
            return

        self.index = 0
        self.update_info_label(filtered=True)
        self.update_dropdown_counts()

        # Save search text + regex flag
        if query:
            new_entry = {"text": query, "regex": self.regex_enabled.get()}
            if new_entry not in self.search_history:
                self.search_history.insert(0, new_entry)
                self.search_box['values'] = [entry['text'] for entry in self.search_history]
                self.save_search_history()

        self.load_record()

    def clear_filter(self):
        self.search_var.set("")  # ✅ Only clear the search input
        self.index = 0  # Reset index to the first record
        self.apply_filter()  # Reapply the filter with current dropdowns
        self.text_body.tag_remove("highlight", "1.0", tk.END)

    def update_info_label(self, filtered=False):
        total = len(self.filtered_df)
        style = "Filtered.TLabel" if filtered or (len(self.filtered_df) != len(self.df)) else "Info.TLabel"
        self.info_label.config(
            text=f"Record {self.index + 1} of {total}",
            style=style
        )

    def start_scrolling(self, direction):
        self.scroll_mode = direction
        self.scroll_step()  # Do the first step immediately

    def scroll_step(self):
        if self.scroll_mode == "next" and self.index < len(self.filtered_df) - 1:
            self.index += 1
            self.load_record()
        elif self.scroll_mode == "prev" and self.index > 0:
            self.index -= 1
            self.load_record()

        # Schedule next step
        self.scroll_job = self.root.after(self.scroll_delay, self.scroll_step)

    def stop_scrolling(self):
        if self.scroll_job:
            self.root.after_cancel(self.scroll_job)
            self.scroll_job = None
            self.scroll_mode = None


def get_unique_value_counts(df, column_name):
    """
    Returns a DataFrame with distinct values and their counts (including NaN) for a given column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The column to analyze.

    Returns:
        pd.DataFrame: A DataFrame with 'value' and 'count' columns.
    """
    value_counts = df[column_name].value_counts(dropna=False)
    result_df = value_counts.reset_index()
    result_df.columns = [column_name, "count"]
    return result_df


# Run the app
root = tk.Tk()
app = TextComparerApp(root, df)
root.mainloop()
