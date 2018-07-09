import glob
import moviepy.editor as mpy


def make_gif(name, fps=6):
    file_list = glob.glob('*.png')  # Get all the pngs in the current directory
    list.sort(file_list, key=lambda x: int(
        x.split('_')[1].split('.png')[0]))  # Sort the images by #, this may need to be tweaked for your use case
    clip = mpy.ImageSequenceClip(file_list, fps=fps)

    exist_list = glob.glob('%s*' % (name))
    if len(exist_list) > 0:
        if len(exist_list) == 1:
            gif_name = '%s_1' % (name)
        else:
            exist_list = glob.glob('%s_*' % (name))
            suffix = map(lambda x: int(x.split('_')[-1].split('.gif')[0]), exist_list)
            gif_name = '%s_%d' % (name, (max(suffix) + 1))
    else:
        gif_name = name

    clip.write_gif('{}.gif'.format(gif_name), fps=fps)
