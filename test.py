

if __name__ == '__main__':
   print("Horse")

   # Assign the filename
   filename = "languages.txt"
   # Open file for writing
   fileHandler = open(filename, "w")

   # Add some text
   fileHandler.write("Bash\n")
   fileHandler.write("Python\n")
   fileHandler.write("PHP\n")

   # Close the file
   fileHandler.close()
