"""
REPLACE ALL $...$ WITH \\(...\\)
"""

import os
import pandas as pd
import re

dir_base = os.getcwd()
dir_posts = os.path.join(dir_base, '..', '..', '_posts')
dir_output = os.path.join(dir_base, 'output')
if not os.path.exists(dir_output):
    os.mkdir(dir_output)

fn_posts = pd.Series(os.listdir(dir_posts))
fn_posts = fn_posts[fn_posts.str.contains('^[0-9]')].to_list()

for ii, fn in enumerate(fn_posts):
    print('Post: %s (%i of %i)' % (fn, ii+1, len(fn_posts)))
    connection = open(os.path.join(dir_posts, fn), 'r')
    lines = pd.Series(connection.readlines())
    connection.close()
    # Remove double $ dollars temporarily
    lines = lines.str.replace('\\${2}', '@@')
    # Replace dollar signs
    # if len(lines[lines.str.contains('\\\\\\$')].to_list()) >0 :
    #     break
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
    copy = open(os.path.join(dir_output, fn), 'w')
    for line in lines:
        copy.write(line)
    copy.close()
