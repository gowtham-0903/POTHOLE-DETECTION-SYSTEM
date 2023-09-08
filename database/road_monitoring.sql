-- phpMyAdmin SQL Dump
-- version 2.11.6
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Sep 13, 2022 at 11:56 AM
-- Server version: 5.0.51
-- PHP Version: 5.2.6

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `road_monitoring`
--

-- --------------------------------------------------------

--
-- Table structure for table `admin`
--

CREATE TABLE `admin` (
  `username` varchar(20) NOT NULL,
  `password` varchar(20) NOT NULL,
  `email` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `admin`
--

INSERT INTO `admin` (`username`, `password`, `email`) VALUES
('admin', 'admin', 'rndittrichy@gmail.com');

-- --------------------------------------------------------

--
-- Table structure for table `rm_report`
--

CREATE TABLE `rm_report` (
  `id` int(11) NOT NULL,
  `filename` varchar(30) NOT NULL,
  `lat` varchar(20) NOT NULL,
  `lon` varchar(20) NOT NULL,
  `rdate` varchar(20) NOT NULL,
  `dtime` timestamp NOT NULL default CURRENT_TIMESTAMP on update CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `rm_report`
--

INSERT INTO `rm_report` (`id`, `filename`, `lat`, `lon`, `rdate`, `dtime`) VALUES
(1, 'aj8i2-ph.jpg', '10.1639', '79.6599', '13-09-2022', '2022-09-13 16:21:36'),
(2, 'bxs4rt-ph.jpg', '10.7842', '78.2267', '13-09-2022', '2022-09-13 17:21:49'),
(3, '5rt6y-ph.jpg', '10.3125', '79.5318', '13-09-2022', '2022-09-13 17:22:40');

-- --------------------------------------------------------

--
-- Table structure for table `rm_upload`
--

CREATE TABLE `rm_upload` (
  `id` int(11) NOT NULL,
  `uname` varchar(20) NOT NULL,
  `filename` varchar(50) NOT NULL,
  `lat` varchar(20) NOT NULL,
  `lon` varchar(20) NOT NULL,
  `location` varchar(50) NOT NULL,
  `rdate` varchar(20) NOT NULL,
  `reply` varchar(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `rm_upload`
--

INSERT INTO `rm_upload` (`id`, `uname`, `filename`, `lat`, `lon`, `location`, `rdate`, `reply`) VALUES
(1, 'raj', '5.jpg', '13.1262', '80.2335', 'SS Road, Trichy', '10-09-2022', 'solved'),
(2, 'raj', '1.jpg', '10.813214', '78.710165', 'Trichy', '12-09-2022', ''),
(3, 'dinesh', 'cm3ed7u-ph.jpg', '11.668798', '78.140287', 'Salem Bus Stand', '13-09-2022', '');

-- --------------------------------------------------------

--
-- Table structure for table `rm_user`
--

CREATE TABLE `rm_user` (
  `id` int(11) NOT NULL,
  `name` varchar(20) NOT NULL,
  `city` varchar(20) NOT NULL,
  `mobile` bigint(20) NOT NULL,
  `email` varchar(40) NOT NULL,
  `uname` varchar(20) NOT NULL,
  `pass` varchar(20) NOT NULL,
  `rdate` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `rm_user`
--

INSERT INTO `rm_user` (`id`, `name`, `city`, `mobile`, `email`, `uname`, `pass`, `rdate`) VALUES
(1, 'Raj', 'Trichy', 9638627415, 'raj@gmail.com', 'raj', '56789', '10-09-2022'),
(2, 'Dinesh', 'Salem', 9054621096, 'dinesh@gmail.com', 'dinesh', '56789', '13-09-2022');
