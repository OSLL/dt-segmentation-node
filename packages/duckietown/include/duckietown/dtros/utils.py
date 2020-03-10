import gc
import rospy

__rh = None


def get_ros_handler(force=False):
    """
    Note from Andrea D. to whoever sees this function.

    This is the story behind this function. ROS developers intentionally left no pointers for
    the end-users to follow that point to some critical objects used by a ROS Node to talk to
    the ROS Master. This was done so that the end-user does mess with the ROS environment.

    Though, if you look at the class `rospy.impl.masterslave.ROSHandler`, you will find that
    there are some functionalities that were never exposed to the end-user. Some of these are
    really useful, for example, did you know that there is a mechanism in place for subscribing
    to ROS params changes? People have built hacky ways to get around this (see
    dynamic_reconfigure).

    So, what do you do when you know that there is an object in memory but you don't know who,
    among the objects you have access to, has a pointer to it?
    Well, I tried to follow the source code, but I had no luck tracing it down.
    The second thing I tried was to spawn a simple ROS node (which is enough to trigger the
    creationg of the object we care about somewhere in memory), the I created a graph off of all
    the objects in memory, where objects formed nodes and their pointers edges in the graph.
    So I tried to find my way to that single instance of `rospy.impl.masterslave.ROSHandler`
    in memory by looking for paths starting from any reachable nodes and terminating there.
    Guess what? no luck.
    Well, as a final resort, I opted for simply asking the GC to hand me a pointer to that object.
    Did I like doing it this way? No.
    Did it work? Hell, yeah!
    Do I sleep at night knowing what I have done? No!
    Do I keep going back trying to solve in a more elegant way? Every day.

    Until I get there, let's use this though.

    :param force:
    :return:
    """
    global __rh
    if __rh is not None and not force:
        return __rh
    rhs = [
        e for e in gc.get_objects()
        if isinstance(e, rospy.impl.masterslave.ROSHandler)
    ]
    if len(rhs) == 0:
        return None
    if len(rhs) == 1:
        __rh = rhs[0]
    if len(rhs) > 1:
        print('WARNING: Two or more ROS Handlers were found in memory. If you happen to see this, '
              'please, post a message on (please, reopen the issue if closed): '
              'https://github.com/duckietown/dt-ros-commons/issues/24.')
    return __rh


def get_namespace(level):
    node_name = rospy.get_name()
    namespace_comps = node_name.lstrip('/').split('/')
    if level > len(namespace_comps):
        level = len(namespace_comps)
    return '/{:s}'.format('/'.join(namespace_comps[:level]))
