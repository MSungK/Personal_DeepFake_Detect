import glob


if __name__ == '__main__':
    path = 'vggface_224'
    files = glob.glob(path + '/**/*.jpg', recursive=True)
    print(len(files))    