"""
Script that will search through the ~/_posts section and make adjustments to both the math brackets and equation tracking. 

For a jupyter notebook:
1. Navigate to the folder, and run `jupyter nbconvert --to markdown {file}.ipynb 
2. Move the figures into ~/figures folder
3. Move the {file}.md to the ~/_posts folder
4. Rename the ~/_posts/{file}.md to a valid ~/_posts/{YYYY}-{mm}-{dd}-{title}.md
5. Change the header of the markdown file

This script will do the following:

1. Replace the ![png]({file}_files/{name}.png) to <center><p><img src="/figures/{file}.png" width="100%"></p></center>
2. REPLACE \label{eq:X} with \tag{1}\label{eq:X}, allowing \eqref{eq:X} to work properly
3. REPLACE ALL $...$ WITH \\(...\\), allowing kramdown to render the math properly


Example usage
> python3 -m _rmd.fix_kramdown --dmin 2024-06-30

"""

# (i) Process arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dmin', default=None, type=str, help='Minimum date to consider')
args = parser.parse_args()
dmin = args.dmin
print('Post date cut-off: %s' % dmin)

# (ii) Load in other modules
import os
import sys
import pandas as pd

# (iii) Setup path directories
dir_base = os.getcwd()
dir_posts = os.path.join(dir_base, '_posts')
assert os.path.exists(dir_posts), f'could not find folder path: {dir_posts}'
dir_output = os.path.join(dir_base, 'output')
if not os.path.exists(dir_output):
    print(f'Making the output directory: {dir_output} for the first time')
    os.mkdir(dir_output)

# (iv) Find relevant posts
fn_posts = pd.Series(os.listdir(dir_posts))
fn_posts = fn_posts[fn_posts.str.contains('^[0-9]')]
date_posts = fn_posts.str.split(pat='\\-[A-Za-z]', n=1, expand=True)
date_posts = pd.to_datetime(date_posts.iloc[:,0])
yy_mm = pd.to_datetime(dmin)
fn_posts = fn_posts[date_posts >= yy_mm].to_list()
if len(fn_posts) == 0:
    sys.exit('No posts written since: %s' % yy_mm)

def get_user_confirmation():
    print(f'The following posts have been found: {fn_posts}')
    while True:
        user_input = input("Do you want to continue? (y/n): ").strip().lower()
        if user_input == 'y':
            print("Continuing the script...")
            # Place the code to continue your script here
            break
        elif user_input == 'n':
            print("Exiting the script...")
            # Place the code to exit your script here
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

# Example usage
get_user_confirmation()

# (v) Loop over all posts to be converted
for ii, fn in enumerate(fn_posts):
    print('Post: %s (%i of %i)' % (fn, ii+1, len(fn_posts)))
    # sys.exit('tmp stop')

    # (a) Load the file
    connection = open(os.path.join(dir_posts, fn), 'r')
    lines = pd.Series(connection.readlines())
    connection.close()

    # (b)
    # Replace the ![png]({file}_files/{name}.png) to <center><p><img src="/figures/{file}.png" width="100%"></p></center>

    # (c) Find the equations
    eq_labels = lines[lines.str.contains('\\\\label\\{eq\\:')]
    if len(eq_labels) > 0:
        print('Has label/eqref')
        tick = 0
        for kk, eq in eq_labels.iteritems():
            if eq.split('\\label')[0][-1] == ' ':
                print('Needs tag')
                tick += 1
                rep = '\\tag{'+str(tick)+'}\\label'
                lines[kk] = eq.replace('\\label', rep)
            else:
                print('Already has tag')
    # Remove double $ dollars temporarily
    lines = lines.str.replace('\\${2}', '@@')
    # Replace currency dollar signs
    lines = lines.str.replace('\\\\\\$', '~~')

    should_skip = False
    for jj, line in enumerate(lines):
        # ignore until we find
        if ('{%' in line) | ('blockquote' in line) | ('```' in line):
            should_skip = not should_skip
            #print('skipping: %s, line: %s' % (should_skip,line))
        if should_skip:
            continue
        if '$' in line:
            ss = line.split('$')
            n = len(ss)
            assert n % 2 == 1
            assert n >= 3
            tmp = ss[0]
            for idx in range(1, n):
                if idx % 2 == 1:
                    tmp += '\\\\(' + ss[idx]
                else:
                    tmp += '\\\\)' + ss[idx]
            print('--------Replacement for: %i-----------' % jj)
            print('Original: %s\nNew: %s' % (line, tmp))
            lines[jj] = tmp
    # Under the blocks
    lines = lines.str.replace('@@','$$')
    lines = lines.str.replace('~~', '\\\\$')
    # Now write
    path_w = os.path.join(dir_output,fn)
    copy = open(path_w, 'w')
    for line in lines:
        copy.write(line)
    copy.close()
