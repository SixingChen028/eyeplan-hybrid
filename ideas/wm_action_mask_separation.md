# Working-memory action mask separation

The capacity-model follow-up `53a4701` separated action reachability from working-memory contents. A future `wm_decay` change could make the root and current fixation neighbors legal fixation actions without automatically refreshing them into working memory.

Under the current decay model, the fixated node, its parent, its children, and the root are refreshed into working memory. If action-mask separation is introduced, the root would no longer always be in working memory; it would remain legally reachable through the action mask instead.

The same distinction matters for neighbors. With `wm_neighbor_activation = 0`, the parent and children of the current fixation would be legal next actions but would not be refreshed into working memory. This has no practical difference for neighbor refresh when `wm_neighbor_activation > 0`, except that legal reachability would no longer depend on whether the neighbor survived in working memory.

Caveat: current and pre-capacity validation rejects `wm_neighbor_activation = 0`. Making this case observable would require an intentional validation change as part of the model update.
