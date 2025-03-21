import os
import numpy as np
import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import filedialog
import threading
from autocorrelator import autocorrelation_fft

class DragDropApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()


        #----------- basic config -----------

        self.title("Acf Chef")
        self.geometry("600x500")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        

        # -------------- File / Folder section --------------

        self.file_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.file_frame.pack(pady=20, padx=20, expand=True)

        self.frame = ctk.CTkFrame(self.file_frame, width=380, height=60, corner_radius=10) # drag and drop
        self.frame.grid(row=0, column=0, padx=(0, 10), sticky="w")

        self.label = ctk.CTkLabel(self.frame, text="Drag Folder or File Here", width=380)
        self.label.pack(pady=10)

        self.middle_text = ctk.CTkLabel(self.file_frame, text="or", font=("Arial", 14, "bold"))
        self.middle_text.grid(row=0, column=1, padx=10)
        
        self.button = ctk.CTkButton(self.file_frame, text="Select Folder", command=self.select_folder, width=120, height=45)
        self.button.grid(row=0, column=2, sticky="e")

        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.on_drop)

        # Listbox to Show File Names
        self.file_listbox = ctk.CTkTextbox(self.file_frame, height=100, width=350)
        self.file_listbox.grid(row=1, column=0, columnspan=3, padx=10, pady=20)




        # ------------ Converter ------------

        self.main_frame = ctk.CTkFrame(self, fg_color = 'transparent')
        self.main_frame.pack(pady = 20, padx = 20,  expand=True)

        self.c_button = ctk.CTkButton(self.main_frame, text="Start Cooking")
        self.c_button.pack(pady = 10)

        self.c_display = ctk.CTkTextbox(self.main_frame, height = 50, width = 350)
        self.c_display.pack(pady = 10)


        # ------------------- File Manager funcs -----------------

    def on_drop(self, event):
        path = event.data.strip()

        # Remove curly brackets from file path if they exist (Windows quirk)
        if path.startswith("{") and path.endswith("}"):
            path = path[1:-1]

        if os.path.isdir(path):
            # If it's a directory, list the files inside
            self.label.configure(text=f"Folder: {path}")
            self.list_files(path)
        
        elif os.path.isfile(path):
            # If it's a file, just display its name
            self.label.configure(text=f"File: {os.path.basename(path)}")
            self.file_listbox.delete("1.0", "end")  # Clear previous entries
            self.file_listbox.insert("end", path + "\n")
        
        else:
            self.label.configure(text="Invalid File or Folder!")

    def select_folder(self):
        """Opens a folder selection dialog."""
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.label.configure(text=f"Folder: {folder_path}")
            self.list_files(folder_path)

    def list_files(self, folder_path):
        """Lists files from the selected folder."""
        self.file_listbox.delete("0.0", "end")  # Clear previous entries
        files = sorted(
        [f for f in os.listdir(folder_path)],
        key=lambda x: int(''.join(filter(str.isdigit, x))) 
        )
        for file in files:
            self.file_listbox.insert("end", file + "\n")





# ------- Converter Main Funcs ---------------

        def start_acf_processing(self):
                """Starts file processing in a separate thread."""
                if not self.selected_folder:
                    self.c_display.insert("end", "No folder selected!\n")
                    return

        # Seperate thread to avoid freezing
        threading.Thread(target=self.acf_from_binaryfiles, args=(self.selected_folder,), daemon=True).start()


    
    
    def log_output(self, message):
        self.c_display.insert("end", message + "\n")
        self.c_display.see("end")  # Auto-scroll

    def binary_to_arr(self, file_path):
        with open(file_path, "rb") as file:
            binary_data = file.read()

        photon_counts = np.frombuffer(binary_data[9:], dtype=np.uint8)  # Ignore first 9 bytes
        return photon_counts

    def acf_from_binaryfiles(self, folder_path):
        """Processes selected binary files and saves autocorrelation results."""
        file_list = self.file_listbox.get("1.0", "end").strip().split("\n")
        if not file_list or file_list == [""]:
            self.log_output("Error: No valid files found in the list!")
            return

        acf_folder = os.path.join(folder_path, "acf")
        os.makedirs(acf_folder, exist_ok=True)

        for i, filename in enumerate(file_list):
            file_path = os.path.join(folder_path, filename)
            save_path = os.path.join(acf_folder, f"acf_bin_{i}.dat")

            self.log_output(f"Processing {filename}...")

            try:
                sequence = self.binary_to_arr(file_path)

                # Ensure autocorrelation_fft() is defined or imported
                acf_binned, bins = self.autocorrelation_fft(sequence, binning=1.03)

                stacked = np.column_stack((bins, acf_binned))
                np.savetxt(save_path, stacked, fmt='%e')

                self.log_output(f"Saved: {save_path}")

            except Exception as e:
                self.log_output(f"Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    app = DragDropApp()
    app.mainloop()

