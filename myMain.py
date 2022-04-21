from tkinter import *
  
from tkinter import filedialog
import SPATIAL_CFIS
# Function for opening the
# file explorer window
def browseFiles():
    filename = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = (("Text files",
                                                        "*.xml"),
                                                       ("all files",
                                                        "*.*")))
    label_file_explorer.configure(text = filename)

def browseFolder():
    filename = filedialog.askdirectory()
    label_data.configure(text = filename)
def Run():
    config_File = label_file_explorer["text"]
    Data_Folder = label_data["text"]
    SPATIAL_CFIS.Spatial_CFIS(config_File, Data_Folder) 
def main1():                                                                                                  
    # Create the root window
    window = Tk()
    
    # Set window title
    window.title('File Explorer')
    
    # Set window size
    window.geometry("500x500")
    
    #Set window background color
    window.config(background = "white")
    
    # Create a File Explorer label
    label_file_explorer = Label(window,text = "config file",width = 100, height = 4,fg = "blue", border = 1)
    label_data = Label(window,text = "Data Folder",width = 100, height = 4,fg = "blue", border = 1)
    button_explore = Button(window,text = "Browse Files",command = browseFiles)
    button_browse_folder = Button(window, text = "Browse folder", command = browseFolder)
    button_run = Button(window, text = "Run", commapythond = Run)
    button_exit = Button(window, text = "Exit", command = exit)
    label_file_explorer.grid(column = 1, row = 1)
    button_explore.grid(column = 1, row = 2)
    label_data.grid(column = 1, row = 4)  
    button_browse_folder.grid(column = 1, row = 5)
    button_run.grid(column = 1, row = 6)
    button_exit.grid(column = 1,row = 7)
    # Let the window wait for any events
    window.mainloop()
#def main():
if __name__ == '__main__':
    # Create the root window
    window = Tk()
    
    # Set window title
    window.title('File Explorer')
    
    # Set window size
    window.geometry("500x500")
    
    #Set window background color
    window.config(background = "white")
    
    # Create a File Explorer label
    label_file_explorer = Label(window,text = "config file",width = 100, height = 4,fg = "blue", border = 1)
    label_data = Label(window,text = "Data Folder",width = 100, height = 4,fg = "blue", border = 1)
    button_explore = Button(window,text = "Browse Files",command = browseFiles)
    button_browse_folder = Button(window, text = "Browse folder", command = browseFolder)
    button_run = Button(window, text = "Run", command = Run)
    button_exit = Button(window, text = "Exit", command = exit)
    label_file_explorer.grid(column = 1, row = 1)
    button_explore.grid(column = 1, row = 2)
    label_data.grid(column = 1, row = 4)  
    button_browse_folder.grid(column = 1, row = 5)
    button_run.grid(column = 1, row = 6)
    button_exit.grid(column = 1,row = 7)
    # Let the window wait for any events
    window.mainloop()
    
    