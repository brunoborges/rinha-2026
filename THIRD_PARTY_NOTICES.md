# Third-Party Notices

This project includes source code derived from third-party open-source projects.

## gb/jvmoonshot-xxvi

The KD-tree nearest-neighbor index under
`src/main/java/io/github/brunoborges/rinha2026/kdport/` (classes `KdTree`,
`KdTreeLayout`, `KdTreeScratch`, `KdTreeTuning`, `KdTreeBuilder`, `KdTreeUnsafe`,
`KdTreeMmap`, `KdTreeProbes`, `KdTreeIO`, `TopKSortedArray`, `VectorIndex`) is
ported from the `gb/jvmoonshot-xxvi` project, with local changes limited to
package/import rewrites and a heap/mmap distance-kernel adaptation. Each ported
file retains an attribution header.

Original work licensed under the MIT License:

```
MIT License

Copyright (c) 2026 Gabriel Tobias

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
