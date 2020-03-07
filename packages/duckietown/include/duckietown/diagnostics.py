import rospy
import time
from threading import Lock

from .constants import *
from .utils import get_namespace

from duckietown_msgs.msg import \
    DiagnosticsRosTopic,\
    DiagnosticsRosTopicArray,\
    DiagnosticsRosLink, \
    DiagnosticsRosLinkArray, \
    DiagnosticsRosParameters, \
    DiagnosticsRosParameter, \
    DiagnosticsRosParameterArray


class DTROSDiagnostics:

    instance = None

    def __init__(self):
        pass

    @classmethod
    def getInstance(cls):
        if not DIAGNOSTICS_ENABLED:
            return None
        if cls.instance is None:
            cls.instance = _DTROSDiagnosticsManager()
        return cls.instance

    @classmethod
    def enabled(cls):
        return DIAGNOSTICS_ENABLED


class _DTROSDiagnosticsManager:

    def __init__(self):
        # initialize diagnostics data containers
        self._topics_stats = {}
        self._topics_stats_lock = Lock()
        self._links_stats = {}
        self._links_stats_lock = Lock()
        self._params_stats = {}
        self._params_stats_lock = Lock()
        # initialize publishers
        self._topics_diagnostics_pub = rospy.Publisher(
            apply_ns(DIAGNOSTICS_ROS_TOPICS_TOPIC, 1),
            DiagnosticsRosTopicArray,
            queue_size=1
        )
        self._params_diagnostics_pub = rospy.Publisher(
            apply_ns(DIAGNOSTICS_ROS_PARAMETERS_TOPIC, 1),
            DiagnosticsRosParameters,
            queue_size=1
        )
        self._links_diagnostics_pub = rospy.Publisher(
            apply_ns(DIAGNOSTICS_ROS_LINKS_TOPIC, 1),
            DiagnosticsRosLinkArray,
            queue_size=1
        )
        # topics diagnostics timer
        self._topics_diagnostics_timer = rospy.Timer(
            period=rospy.Duration.from_sec(DIAGNOSTICS_ROS_TOPICS_PUB_EVERY_SEC),
            callback=self._publish_topics_diagnostics,
            oneshot=False)
        # parameters diagnostics timer
        self._parameters_diagnostics_timer = rospy.Timer(
            period=rospy.Duration.from_sec(DIAGNOSTICS_ROS_PARAMETERS_PUB_EVERY_SEC),
            callback=self._publish_parameters_diagnostics,
            oneshot=False)
        # links diagnostics timer
        self._links_diagnostics_timer = rospy.Timer(
            period=rospy.Duration.from_sec(DIAGNOSTICS_ROS_LINKS_PUB_EVERY_SEC),
            callback=self._publish_links_diagnostics,
            oneshot=False)

    def register_topic(self, name, direction, healthy_freq, topic_types):
        try:
            self._topics_stats_lock.acquire()
            self._topics_stats[name] = {
                'types': [t.value for t in topic_types],
                'direction': direction.value,
                'frequency': 0.0,
                'healthy_freq': healthy_freq,
                'bandwidth': -1.0,
                'processing_time': -1.0,
                'enabled': True
            }
        finally:
            self._topics_stats_lock.release()

    def register_param(self, name):
        try:
            self._params_stats_lock.acquire()
            self._params_stats[name] = {
                # in the future, we will have param details
            }
        finally:
            self._params_stats_lock.release()

    def set_topic_switch(self, name, switch_status):
        if name in self._topics_stats:
            self._topics_stats[name]['enabled'] = switch_status

    def compute_topics_frequency(self):
        try:
            self._links_stats_lock.acquire()
            # ---
            # topic frequency is computed as the average of the frequencies on all its connections
            for topic in list(self._topics_stats.keys()):
                freq = []
                bwidth = []
                for _, link in self._links_stats.items():
                    if link['topic'] == topic:
                        freq.append(link['frequency'])
                        bwidth.append(link['bandwidth'])
                # compute frequency and bandwidth
                self._topics_stats_lock.acquire()
                self._topics_stats[topic]['frequency'] = \
                    sum(freq) / (len(freq) if len(freq) else 1)
                self._topics_stats[topic]['bandwidth'] = \
                    sum(bwidth) / (len(bwidth) if len(bwidth) else 1)
                self._topics_stats_lock.release()
            # ---
        finally:
            self._links_stats_lock.release()

    def _compute_stats(self):
        # get bus stats and bus info for every active connection
        connections = rospy.impl.registration.get_topic_manager().get_pub_sub_info()
        pub_stats, sub_stats = rospy.impl.registration.get_topic_manager().get_pub_sub_stats()
        # ---
        _conn_direction = lambda d: {
            'i': TopicDirection.INBOUND.value, 'o': TopicDirection.OUTBOUND.value
        }[d]
        # connections stats
        # From (_TopicImpl:get_stats_info):
        #   http://docs.ros.org/melodic/api/rospy/html/rospy.topics-pysrc.html
        connection_info = {
            c[0]: {
                'topic': c[4],
                'remote': c[1],
                'direction': _conn_direction(c[2]),
                'connected': c[5],
                'transport': c[3][:3]
            }
            for c in connections
            if len(c) >= 6
        }
        links = {}
        # publishers stats
        # From (_PublisherImpl:get_stats):
        #   http://docs.ros.org/melodic/api/rospy/html/rospy.topics-pysrc.html
        for pub_stat in pub_stats:
            topic_name, message_data_bytes, conn_stats = pub_stat
            for conn in conn_stats:
                if len(conn) != 4:
                    continue
                _id, _bytes, _num_messages, _ = conn
                if _id not in connection_info:
                    continue
                link_info = {
                    "bytes": _bytes,
                    "messages": _num_messages,
                    "dropped": 0,
                    "_time": time.time()
                }
                link_info.update(connection_info[_id])
                # compute frequency
                if _id in self._links_stats:
                    old_reading = self._links_stats[_id]
                    link_info.update(_compute_f_b(link_info, old_reading))
                else:
                    link_info.update({'frequency': 0.0, 'bandwidth': 0.0})
                links[_id] = link_info
        # subscribers stats
        # From (_SubscriberImpl:get_stats):
        #   http://docs.ros.org/melodic/api/rospy/html/rospy.topics-pysrc.html
        for sub_stat in sub_stats:
            topic_name, conn_stats = sub_stat
            for conn in conn_stats:
                if len(conn) != 5:
                    continue
                _id, _bytes, _num_messages, _drop_estimate, _ = conn
                if _id not in connection_info:
                    continue
                link_info = {
                    "bytes": _bytes,
                    "messages": _num_messages,
                    "dropped": _drop_estimate if _drop_estimate > 0 else 0,
                    "_time": time.time()
                }
                link_info.update(connection_info[_id])
                # compute frequency
                if _id in self._links_stats:
                    old_reading = self._links_stats[_id]
                    link_info.update(_compute_f_b(link_info, old_reading))
                else:
                    link_info.update({'frequency': 0.0, 'bandwidth': 0.0})
                links[_id] = link_info
        # ---
        # update link stats
        try:
            self._links_stats_lock.acquire()
            self._links_stats = links
        finally:
            self._links_stats_lock.release()
        # update topic stats
        self.compute_topics_frequency()

    def update_topic_processing_time(self, name, proc_time):
        if name in self._topics_stats:
            lpt = self._topics_stats[name]['processing_time']
            if lpt > 0:
                # average the previous value (weight = 40%) with the new one (weight = 60%)
                self._topics_stats[name]['processing_time'] = 0.4 * lpt + 0.6 * proc_time
            else:
                self._topics_stats[name]['processing_time'] = proc_time

    def _publish_topics_diagnostics(self, *args, **kwargs):
        msg = DiagnosticsRosTopicArray()
        msg.header.stamp = rospy.Time.now()
        try:
            self._topics_stats_lock.acquire()
            for topic, topic_stats in self._topics_stats.items():
                msg.topics.append(DiagnosticsRosTopic(
                    node=rospy.get_name(),
                    topic=topic,
                    **topic_stats
                ))
        finally:
            self._topics_stats_lock.release()
        self._topics_diagnostics_pub.publish(msg)

    def _publish_parameters_diagnostics(self, *args, **kwargs):
        msg = DiagnosticsRosParameterArray()
        msg.header.stamp = rospy.Time.now()
        try:
            self._params_stats_lock.acquire()
            for param, param_stats in self._params_stats.items():
                msg.params.append(DiagnosticsRosTopic(
                    node=rospy.get_name(),
                    param=param,
                    **param_stats
                ))
        finally:
            self._params_stats_lock.release()
        self._params_diagnostics_pub.publish(msg)

    def _publish_links_diagnostics(self, *args, **kwargs):
        self._compute_stats()
        msg = DiagnosticsRosLinkArray()
        msg.header.stamp = rospy.Time.now()
        try:
            self._links_stats_lock.acquire()
            for _, link_stats in self._links_stats.items():
                msg.links.append(DiagnosticsRosLink(
                    node=rospy.get_name(),
                    **{k: v for k, v in link_stats.items() if not k.startswith('_')}
                ))
        finally:
            self._links_stats_lock.release()
        self._links_diagnostics_pub.publish(msg)


def _compute_f_b(new_read, old_read):
    tnow = time.time()
    return {
        'frequency': (new_read['messages'] - old_read['messages']) / (tnow - old_read['_time']),
        'bandwidth': (new_read['bytes'] - old_read['bytes']) / (tnow - old_read['_time'])
    }


def apply_ns(name, ns_level):
    return '{:s}/{:s}'.format(
        get_namespace(ns_level).rstrip('/'),
        name.strip('/')
    )
