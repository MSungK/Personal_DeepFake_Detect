import glob


if __name__ == '__main__':
    path = 'vggface2-HQ/VGGface2_None_norm_512_true_bygfpgan'
    files = glob.glob(path + '/**/*.jpg', recursive=True)
    print(len(files))    